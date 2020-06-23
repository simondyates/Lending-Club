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
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from Portfolio import Portfolio
from YieldCurve import YieldCurve
import plotly.graph_objects as go
import plotly.io as pio
import psutil

# Global parameters
lc_fee = .01 # i.e. 1%.  This is charged on all on-time payments.  I'm assuming the numbers we have are gross so we need to subtract this
rec_lag = 9 # recoveries arrive 'rec_lag' months after last_pymnt_d
first_invest = 20100101 # Use data before this point to fit the starting model.  This gives us 1 year of data (not a lot).
refit = 6 # Refit the model every 'refit' months
est_steps = 10 # Add this many estimators to the model each refit

# Tactic parameters
max_grade = 14 # i.e. Grades A-D
max_per_loan = 1000 # USD in any one loan
max_per_month = 100000 / 6 # USD max spend in any one month - to slow speed of initial ramp up
max_funding = 100000 # USD. The maximum amount of external funding that's available
min_ret = .1 # The minimum required value of (PV/Principal-1) in order to invest in a loan


########################################################################################################
#  Execution part of codebase
########################################################################################################

# Read in data
accept = pd.read_pickle('../derivedData/train.pkl')
# Remove rows not meeting grade criteria
accept = accept[accept['sub_grade'] <= max_grade]
# set up the Yield Curve
rates = pd.read_csv('../macroData/Rates.csv', index_col=0)
rates.index = pd.to_datetime(rates.index)
yc = YieldCurve(rates)

# Split target from attributes and normalise attribs
y = (accept['PV'] / accept['funded_amnt']).to_numpy()
X = accept.drop(['id', 'PV', 'loan_status'], axis=1)

# Drop attributes with updates after loan inception
leaks = ['recoveries', 'total_pymnt', 'dti', 'last_pymnt_d',
         'revol_util', 'open_acc', 'pub_rec', 'revol_bal',
         'revol_util', 'delinq_2yrs', 'RF_rate']
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
port_sel = Portfolio(accept, yc, rec_lag, lc_fee) # will track the best loans we select
port_rand = Portfolio(accept, yc, rec_lag, lc_fee) # will track a randomly-selected portfolio

# Run the sim for each month that loans were issued
dates = np.sort(X.loc[X['issue_d'] >= first_invest, 'issue_d'].unique())

for i, date in enumerate(dates):
    if (i % refit) == 0:
        print(f'Refitting for {date}')
        X_train = X[X['issue_d'] < date]
        y_train = y[X['issue_d'] < date]
        X_train_s = ct.fit_transform(X_train)
        X_s = ct.transform(X)
        boost.set_params(n_estimators= 35 + int( (i / refit) * est_steps) )
        boost.fit(X_train_s, y_train)
        print(f'IS R^2: {boost.score(X_train_s, y_train):.2%}')
    # Determine what we have to invest this month
    if str(date)[4:6] == '01':
        print(f'Simulating for {date}')
    d = dt.datetime.strptime(str(date), '%Y%m%d')
    budget = min(max_per_month, max_funding + port_sel.spent(d))
    n_to_buy = int(budget / max_per_loan)
    # Select the loans that will issue this month
    bools = X['issue_d']==date
    idx = X.index[bools]
    # Predict PVs for them
    rets = pd.Series(boost.predict(X_s[bools, :]), index=idx).sort_values()
    # Purchase the selected and random loans
    bought = len([port_sel.add(rets.index[-i], max_per_loan)
                  for i in range(1, n_to_buy + 1) if rets.iloc[-i] >= min_ret])
    [port_rand.add(ix, max_per_loan)
         for ix in np.random.choice(idx, bought, replace=False)]

# Generate results
fv_date = max(port_sel.receipts.keys())
res_sel = port_sel.get_cashflows(fv_date)
res_rand = port_rand.get_cashflows(fv_date)
results = res_sel.join(res_rand, how='outer', lsuffix='_sel', rsuffix='_rand')
results['PVs_sel'] = results['PVs_sel'] / results['n_loans_sel']
results['PVs_rand'] = results['PVs_rand'] / results['n_loans_rand']
offered = accept[['issue_d', 'funded_amnt']].copy()
offered['issue_d'] = offered['issue_d'].apply(lambda i: dt.datetime.strptime(str(i), '%Y%m%d'))
offered = offered.groupby('issue_d').agg(['count', 'sum'])
offered.columns = offered.columns.get_level_values(1)
results = results.join(offered)
results['pct_of_loans'] = results['n_loans_sel'] / results['count']
results['pct_of_not'] = -results['payments_sel'] / results['sum']
results['year'] = results.index.map(lambda d: d.year)
results = results.groupby('year').agg({'n_loans_sel': sum, 'pct_of_loans': np.mean, 'pct_of_not': np.mean,
                                       'payments_sel': sum, 'receipts_sel': sum, 'net_sel': sum,
                                       'receipts_rand': sum, 'net_rand': sum, 'PVs_sel': np.mean, 'PVs_rand': np.mean})

# Output results
print('-'*25)
print(f'Total funded: {port_sel.drawdown(fv_date):,.0f}')
print(f'Total FV of selected: {port_sel.FV(fv_date):,.0f}')
print(f'Total FV of rand: {port_rand.FV(fv_date):,.0f}')
print(f'IRR of selected: {port_sel.IRR(fv_date):.1%}')
print(f'IRR of rand: {port_rand.IRR(fv_date):.1%}')
print('-'*25)

pio.renderers.default = "browser"
results = results.reset_index()
heads = ['Year', '# Loans', '% LC by Count', '% LC by Value', 'Payments', 'Receipts (selected)', 'Net (selected)',
         'Receipts (random)', 'Net (random)', 'PV/loan (selected)', 'PV/loan (random)']
fig = go.Figure(data=[go.Table(
    header=dict(values=heads,
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[results[col] for col in results.columns],
               fill_color='lavender',
               align='right',
               format = [[None], [None], ['.1%'], ['.1%'],
                        [',.0f'], [',.0f'], [',.0f'], [',.0f'],
                        [',.0f'], [',.0f'], [',.0f']]))
])
fig.update_layout(title=f'IRR of Selected Loans {port_sel.IRR(fv_date):.1%} with Average Rating {port_sel.rating()} vs. {port_rand.IRR(fv_date):.1%} for Random')
fig.show()
n = dt.datetime.today()
t_str = dt.datetime.strftime(n, '%H%M%S')
fig.write_image('../rawData/Sim' + t_str + '.jpg', width=1000, height=600)

# Tweak the model: hyper param tuning and pruning
# Fit tactic params