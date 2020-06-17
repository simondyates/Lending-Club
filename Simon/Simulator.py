# Simulate a trading strategy as follows:
# Review each loan chronologically and invest if predicted PV > loan_amnt * (1 + min_ret)
# Continue until the full amount (say $100,000) has been invested, respecting monthly maximum investments
# Then, purchase new loans meeting PV criteria only to reinvest cashflows received

########################################################################################################
#  Preamble: imports; global and tactic params; and Portfolio class definition
########################################################################################################

import pandas as pd
import datetime as dt
import numpy as np
import numpy_financial as npf
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from collections import defaultdict

# Global parameters
lc_fee = .01 # i.e. 1%.  This is charged on all on-time payments.  I'm assuming the numbers we have are gross so we need to subtract this
rec_lag = 9 # recoveries arrive 'rec_lag' months after last_pymnt_d
first_invest = 20100101 # Use data before this point to fit the starting model.  This gives us 1 year of data (not a lot).
refit = 24 # Refit the model every 'refit' months
est_steps = 30 # Add this many estimators to the model each refit

# Tactic parameters
max_per_loan = 1000 # USD in any one loan
max_per_month = 100000 / 6 # USD max spend in any one month - to slow speed of initial ramp up
max_portfolio = 100000 # USD. The current implementation interprets this as max external funding requirement
min_ret = .1 # The minimum required value of (PV/Principal-1) in order to invest in a loan

# Define a class to track a portfolio of LC loans; construct cash-flows for them; and calculate metrics
class Portfolio:
    def __init__(self, universe, rates, rec_lag=0):
        self.__univ = universe # df of all the loan data we have
        self.__rates = rates # df of historical risk-free rates
        self.rec_lag = rec_lag
        self.ids = []
        self.amnts = []
        self.payments = defaultdict(int)
        self.receipts = defaultdict(int)

    def __gen_cfs__(self, id, amnt):
        start = dt.datetime.strptime(self.__univ.loc[id, 'issue_d'].astype('str'), '%Y%m%d')
        stop = dt.datetime.strptime(self.__univ.loc[id, 'last_pymnt_d'].astype('str'), '%Y%m%d')
        self.payments[start] -= amnt
        ratio = amnt / self.__univ.loc[id, 'funded_amnt']
        months = int((stop - start).days * 12 / 365)
        # adjust payments for LC's 1% fee
        pmts = ratio * (self.__univ.loc[id, 'total_pymnt'] - self.__univ.loc[id, 'recoveries']) * (1 - lc_fee)
        if months == 0:
            self.receipts[start] += pmts
        else:
            d = start
            for i in range(1, months):
                d = d + dt.timedelta(days=31)
                d = dt.datetime(d.year, d.month, 1)
                self.receipts[d] += pmts / months
        recs = ratio * self.__univ.loc[id, 'recoveries'] # No LC fee on recoveries
        if recs > 0:
            d = stop + dt.timedelta(days = 31*rec_lag)
            d = dt.datetime(d.year, d.month, 1)
            self.receipts[d] += recs

    def add(self, id, amnt):
        self.ids.append(id)
        self.amnts.append(amnt)
        self.__gen_cfs__(id, amnt)

    def get_cashflows(self, d):
        dates = sorted(set(self.payments.keys()).union(set(self.receipts.keys())))
        dates = [dat for dat in dates if dat <= d]
        if dates == []:
            df1 = pd.DataFrame({'payments': 0, 'receipts': 0, 'net': 0}, index=[d])
        else:
            s1 = pd.Series([self.payments[dat] for dat in dates if dat <= d], index=dates, name='payments')
            s2 = pd.Series([self.receipts[dat] for dat in dates if dat <= d], index=dates, name='receipts')
            df1 = pd.concat([s1, s2], axis=1)
            df1['net'] = df1['payments'] + df1['receipts']
        return(df1)

    def spent(self, d):
        # returns total net cashflows for the investor to date d
        df2 = self.get_cashflows(d)
        return(df2['net'].sum())

    def drawdown(self, d):
        # returns the worst of the cumulative net cashflows for the investor up to date d
        df3 = self.get_cashflows(d)
        df3['cum_sum'] = df3['net'].cumsum()
        return (df3['cum_sum'].min())

    def FV(self, d):
        # future value of net cashflows to date d
        df4 = self.get_cashflows(d)
        df4['days'] = (d - df4.index).days

        def get_rate(term, as_of):
            df = self.__rates
            idx = df.index.get_loc(as_of, method='pad')
            if term <= 15:
                r = df.iloc[idx, 6]  # FF
            elif term <= 365:
                r = df.iloc[idx, 5]  # 3mo LIBOR
            elif term <= 365 * 2.5:
                r = df.iloc[idx, 2]  # 2yr T
            elif term <= 365 * 4:
                r = df.iloc[idx, 3]  # 3yr T
            else:
                r = df.iloc[idx, 4]  # 5yr T
            return(r)

        df4['dates'] = df4.index
        df4['rates'] = df4[['days', 'dates']].apply(lambda x: get_rate(x[0], x[1]), axis=1)
        # Assume semi-annual compounding for all rates (true for the T rates, fudge for money-market)
        FV = df4['net'] * (1 + df4['rates']/200) ** (2*df4['days']/365)
        return(FV.sum())

    def IRR(self, d):
        # IRR of net cashflows to date d
        df5 = self.get_cashflows(d)
        r = npf.irr(df5['net'])
        return(2 * ((1 + r)**6 - 1))

########################################################################################################
#  Execution part of codebase
########################################################################################################

# Read in data
accept = pd.read_pickle('../derivedData/train.pkl')
rates = pd.read_csv('../macroData/rates.csv', index_col=0)
rates.index = pd.to_datetime(rates.index)

# Split target from attributes and normalise attribs
y = accept['PV'].to_numpy()
X = accept.drop(['PV', 'loan_status'], axis=1)

# Drop attributes with updates after loan inception
leaks = ['recoveries', 'total_pymnt', 'dti', 'last_pymnt_d',
         'revol_util', 'open_acc', 'pub_rec', 'revol_bal',
         'revol_util', 'delinq_2yrs']
X = X.drop(leaks, axis=1)

# Initialise scaler
date_cols = ['issue_d','earliest_cr_line']
num_cols = ['loan_amnt', 'funded_amnt', 'term', 'int_rate', 'grade',
            'sub_grade', 'emp_length', 'home_ownership', 'annual_inc',
            'fico_range_high', 'fico_range_low']
cols_to_scale = date_cols + num_cols
ct = make_column_transformer((StandardScaler(), cols_to_scale), remainder='passthrough')

# Initialise the model and two empty portfolios
boost = GradientBoostingRegressor(subsample=0.1, verbose=0, warm_start=False)
port_sel = Portfolio(accept, rates, rec_lag) # will track the best loans we select
port_rand = Portfolio(accept, rates, rec_lag) # will track a randomly-selected portfolio

# Run the sim for each month that loans were issued
dates = np.sort(X.loc[X['issue_d'] >= first_invest, 'issue_d'].unique())

for i, date in enumerate(dates):
    if (i % refit) == 0:
        print(f'Refitting for {date}')
        X_train = X[X['issue_d'] < date]
        y_train = y[X['issue_d'] < date]
        X_train_s = ct.fit_transform(X_train)
        X_s = ct.transform(X)
        boost.set_params(n_estimators= 35 + int(i * est_steps / 12))
        boost.fit(X_train_s, y_train)
        print(f'IS R^2: {boost.score(X_train_s, y_train):.2%}')
    # Determine what we have to invest this month
    d = dt.datetime.strptime(str(date), '%Y%m%d')
    budget = min(max_per_month, max_portfolio + port_sel.spent(d))
    n_to_buy = int(budget / max_per_loan)
    # Select the loans that will issue this month
    bools = X['issue_d']==date
    idx = X.index[bools]
    # Predict PVs for them
    pvs = boost.predict(X_s[bools, :])
    amnts = X.loc[idx, 'funded_amnt']
    # Calculate return ratios
    rets = (pvs / amnts).sort_values()
    # Purchase the selected and random loans
    bought = len([port_sel.add(rets.index[-i], max_per_loan)
                  for i in range(1, n_to_buy + 1) if rets.iloc[-i] >= min_ret])
    [port_rand.add(ix, max_per_loan)
         for ix in np.random.choice(idx, bought, replace=False)]

fv_date = max(port_sel.receipts.keys())
print('-'*25)
print(f'Total funded: {port_sel.drawdown(fv_date):,.0f}')
print(f'Total FV of selected: {port_sel.FV(fv_date):,.0f}')
print(f'Total FV of rand: {port_rand.FV(fv_date):,.0f}')
print(f'IRR of selected: {port_sel.IRR(fv_date):.1%}')
print(f'IRR of rand: {port_rand.IRR(fv_date):.1%}')

# Need to add more metrics:
# - what % of loans on offer did we buy (by count and by notional)
# - what was our annual trading P&L (i.e. PV - spent)
# Tweak the model: hyper param tunning and pruning
# Fit tactic params