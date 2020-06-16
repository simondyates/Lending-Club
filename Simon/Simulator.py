# Simulate a trading strategy as follows:
# Review each loan chronologically and invest if predicted PV > loan_amnt * (1 + min_ret)
# Continue until the full amount ($100,000) has been invested, respecting monthly maximum investments
# Then, purchase new loans meeting PV criteria only to reinvest cashflows received

import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
import datetime as dt
import numpy as np

# Global parameters
rec_lag = 9

# Tactic parameters
max_per_loan = 1000
max_per_month = 100000 / 6
max_portfolio = 100000
min_ret = .1

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

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
ct = load('../derivedData/boost_scaler.joblib')
X_train_s = ct.transform(X_train)
X_test_s = ct.transform(X_test)

class Portfolio:
    def __init__(self, universe, rates, rec_lag=0):
        self.__univ = universe
        self.__rates = rates
        self.ids = []
        self.amnts = []
        self.cashflows = {}
        self.rec_lag = rec_lag

    def __gen_cfs(self, id, amnt):
        start = dt.datetime.strptime(self.__univ.loc[id, 'issue_d'].astype('str'), '%Y%m%d')
        stop = dt.datetime.strptime(self.__univ.loc[id, 'last_pymnt_d'].astype('str'), '%Y%m%d')
        if start in self.cashflows:
            self.cashflows[start] -= amnt
        else:
            self.cashflows[start] = amnt
        ratio = amnt / self.__univ.loc[id, 'funded_amnt']
        months = int((stop - start).days * 12 / 365)
        pmts = ratio * (self.__univ.loc[id, 'total_pymnt'] - self.__univ.loc[id, 'recoveries'])
        if months == 0:
            self.cashflows[start] += pmts
        else:
            d = start
            for i in range(1, months):
                d = d + dt.timedelta(days=31)
                d = dt.datetime(d.year, d.month, 1)
                if d in self.cashflows:
                    self.cashflows[d] += pmts / months
                else:
                    self.cashflows[d] = pmts / months
        recs = ratio * self.__univ.loc[id, 'recoveries']
        if recs > 0:
            d = stop + dt.timedelta(days = 31*rec_lag)
            d = dt.datetime(d.year, d.month, 1)
            if d in self.cashflows:
                self.cashflows[d] += recs
            else:
                self.cashflows[d] = recs

    def add(self, id, amnt):
        self.ids.append(id)
        self.amnts.append(amnt)
        self.__gen_cfs(id, amnt)

    def spent(self, d):
        return (sum([self.cashflows[dat] for dat in self.cashflows.keys() if dat <= d]))

    def drawdown(self, d):
        dates = [dat for dat in self.cashflows.keys() if dat <= d]
        drawdowns = [sum([self.cashflows[d] for d in dates[:i]]) for i in range(1, len(dates))]
        if len(drawdowns) == 0:
            return(0)
        else:
            return(min(drawdowns))

    def FV(self, d):
        dates = [dat for dat in self.cashflows.keys() if dat <= d]
        df = self.__rates
        FV = 0
        for dat in dates:
            days = (d - dat).days
            idx = df.index.get_loc(dat, method='pad')
            if days <= 15:
                r = df.iloc[idx, 6]  # FF
            elif days <= 365:
                r = df.iloc[idx, 5]  # 3mo LIBOR
            elif days <= 365 * 2.5:
                r = df.iloc[idx, 2]  # 2yr T
            elif days <= 365 * 4:
                r = df.iloc[idx, 3]  # 3yr T
            else:
                r = df.iloc[idx, 4]  # 5yr T
            # Assume semi-annual compounding
            FV += self.cashflows[dat] * (1 + r/200) ** (2*days/365)
        return(FV)

boost = load('../derivedData/boost.joblib')
port_sel = Portfolio(accept, rates, rec_lag)
port_rand = Portfolio(accept, rates, rec_lag)

dates = np.sort(X_train['issue_d'].unique())
for date in dates:
    # Determine what we have to invest this month
    d = dt.datetime.strptime(str(date), '%Y%m%d')
    budget = min(max_per_month, max_portfolio + port_sel.spent(d))
    n_to_buy = int(budget / max_per_loan)
    # Select the loans that will issue this month
    bools = X_train['issue_d']==date
    idx = X_train.index[bools]
    # Predict PVs for them
    pvs = boost.predict(X_train_s[bools, :])
    amnts = X_train.loc[idx, 'funded_amnt']
    # Calculate return ratios
    rets = (pvs / amnts).sort_values()
    bought = len([port_sel.add(rets.index[-i], max_per_loan)
                  for i in range(1, n_to_buy + 1) if rets.iloc[-i] >= min_ret])
    [port_rand.add(ix, max_per_loan)
     for ix in np.random.choice(idx, bought, replace=False)]

fv_date = max(port_sel.cashflows.keys())
print(f'Total funded: {port_sel.drawdown(fv_date):,.0f}')
print(f'Total FV of selected: {port_sel.FV(fv_date):,.0f}')
print(f'Total FV of rand: {port_rand.FV(fv_date):,.0f}')

# get rid of unused vars in class