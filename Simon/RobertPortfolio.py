import pandas as pd
import datetime as dt
import numpy as np
from Portfolio import Portfolio
from YieldCurve import YieldCurve
import plotly.graph_objects as go
import plotly.io as pio

# Global variables
rec_lag = 9
lc_fee = 0.01

model = 'ridge'
predicts = pd.read_csv('../Robert/optimal_weights_' + model + '.csv', index_col=0)
accept = pd.read_pickle('../derivedData/train.pkl')
accept.set_index('id', inplace=True)
rates = pd.read_csv('../macroData/Rates.csv', index_col=0)
rates.index = pd.to_datetime(rates.index)
yc = YieldCurve(rates)

port_rob = Portfolio(accept, yc, rec_lag, lc_fee)
port_rand = Portfolio(accept, yc, rec_lag, lc_fee)

for loan_id in predicts.index:
    port_rob.add(loan_id, predicts.loc[loan_id, 'weights'] * 100000)
    univ = accept[accept['issue_d'] == accept.loc[loan_id, 'issue_d']]
    port_rand.add(np.random.choice(univ.index, 1)[0], 1000)
    
# Generate results
fv_date = max(port_rob.receipts.keys())
res_sel = port_rob .get_cashflows(fv_date)
res_rand = port_rand .get_cashflows(fv_date)
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
print(f'Total funded: {port_rob.drawdown(fv_date):,.0f}')
print(f'Total FV of selected: {port_rob.FV(fv_date):,.0f}')
print(f'Total FV of rand: {port_rand.FV(fv_date):,.0f}')
print(f'IRR of selected: {port_rob.IRR(fv_date):.1%}')
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
if model == 'randomF':
    fig.update_layout(title=f'IRR of RF Selected Loans {port_rob.IRR(fv_date):.1%} vs. {port_rand.IRR(fv_date):.1%} for Random')
else:
    fig.update_layout(
        title=f'IRR of Ridge Selected Loans {port_rob.IRR(fv_date):.1%} vs. {port_rand.IRR(fv_date):.1%} for Random')
fig.show()

