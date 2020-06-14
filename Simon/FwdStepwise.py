# Runs a forward stepwise multilinear regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from ModelScore import model_score, model_features
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Global Parameters
use_dum = False
use_log = True
scale = True

# import the data
if use_dum:
    house = pd.read_csv('../derivedData/train_cleaned.csv', index_col='Id')
else:
    house = pd.read_csv('../derivedData/train_NotDum.csv', index_col='Id')
house['logSalePrice'] = np.log(house['SalePrice'])

if not (use_dum):
    # Use MV encoding on nominals
    cols_to_enc = house.columns[house.dtypes == 'object']
    for col in cols_to_enc:
        if use_log:
            gp = house.groupby(col)['logSalePrice'].mean()
        else:
            gp = house.groupby(col)['SalePrice'].mean()
        house[col] = house[col].apply(lambda x: gp[x])

# Create train and test sets
X = house.drop(['SalePrice', 'logSalePrice'], axis=1)
cols = X.columns

if use_log:
    y = house['logSalePrice']
else:
    y = house['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if (scale):
    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)

# Initialise Linear model
lin = LinearRegression()

def t_stat(reg, X, y):
    col_bool = reg.coef_ != 0
    X_s = X[:, col_bool]
    sse = np.sum((reg.predict(X) - y) ** 2, axis=0) / float(X_s.shape[0] - X_s.shape[1])
    try:
        xTx_inv = np.linalg.inv(X_s.T @ X_s)
        if min(np.diag(xTx_inv)) <= 0:
            t = np.full([1, X.shape[1]], np.nan)
        else:
            se_s = np.sqrt(np.diagonal(sse * xTx_inv))
            i = 0
            t = np.zeros([1, X.shape[1]])
            for j, b in enumerate(col_bool):
                if not (b):  # i.e. reg.coef is zero
                    t[0, j] = 0
                else:
                    t[0, j] = reg.coef_[j] / se_s[i]
                    i += 1
    except:
        t = np.full([1, X.shape[1]], np.nan)
    return t[0]

def AIC(reg, X, y):
    n = X.shape[0]
    p = sum(reg.coef_ != 0)
    MSE = ((reg.predict(X) - y) ** 2).mean()
    return (n * np.log(MSE) + 2 * (p+1))

# initialise global result variables
col_picks = []
ts = []
ISR2s = []
OSR2s = []
AICs = []
search = [i for i in range(len(cols)) if i not in col_picks]
while len(search) > len(cols) - 20:  # picks the first 20 columns
    print(len(search))
    # initialise result variables for this column set
    col_t = []
    col_ISR2 = []
    for col in search:
        X = X_train[:, col_picks + [col]]
        lin.fit(X, y_train)
        col_t.append(t_stat(lin, X, y_train)[len(col_picks)])
        col_ISR2.append(lin.score(X, y_train))
    # pick the winner
    idx = col_t.index(max(col_t))
    col_picks.append(search[idx])
    ts.append(col_t[idx])
    ISR2s.append(col_ISR2[idx])
    # Use the fully OOS score for the current model
    lin.fit(X_train[:, col_picks], y_train)
    OSR2s.append(lin.score(X_test[:, col_picks], y_test))
    AICs.append(AIC(lin, X_test[:, col_picks], y_test))
    search = [i for i in range(len(cols)) if i not in col_picks]

print(f'Fwd Step train score {lin.score(X_train[:, col_picks], y_train):.02%}')
print(f'Fwd Step test score {lin.score(X_test[:, col_picks], y_test):.02%}')
print(f'Fwd Step cols used {sum(lin.coef_ != 0)}')

# Calculate feature importance
if (scale):
    fwd_feature_imp = pd.Series(abs(lin.coef_), index=cols[col_picks])
else:
    fwd_feature_imp = pd.Series(abs(t_stat(lin, X_train[:, col_picks], y_train)), index=cols).sort_values(
        ascending=False)

# AIC indicates only the first 12 features add value.
lin.fit(X_train[:, col_picks[:12]], y_train)
model_score(lin, X_test[:, col_picks[:12]], y_test, saves=False)
model_features(lin, fwd_feature_imp.index, fwd_feature_imp, saves=False)

# Make a beautiful graph of R^2 vs complexity
results = pd.DataFrame(index=range(len(col_picks)), columns=['col', 'ISR2', 'OSR2', 't_stats', 'Norm AIC'])
results['col'] = cols[col_picks]
results['ISR2'] = ISR2s
results['OSR2'] = OSR2s
results['t_stats'] = ts
if AICs[0] < 0:
    results['Norm AIC'] = (-AICs[0]) / list(map(lambda x:-x, AICs))
else:
    results['Norm AIC'] = AICs / AICs[0]

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=results['col'], y=results['ISR2'], name='In-sample R<sup>2</sup>'))
fig.add_trace(go.Scatter(x=results['col'], y=results['OSR2'], name='Out-sample R<sup>2</sup>'))
fig.add_trace(go.Scatter(x=results['col'], y=results['Norm AIC'], name='AIC (ratio to first model)'))
fig.add_trace(go.Bar(x=results['col'], y=abs(results['t_stats']), name='t stats', opacity=.6), secondary_y=True)
if use_log:
    fig.update_layout(title='R<sup>2</sup>, AIC and t stats for Forward Multilinear Regression of log(SalePrice)')
else:
    fig.update_layout(title='R<sup>2</sup>, AIC and t stats for Forward Multilinear Regression of SalePrice')

fig.update_layout(
    xaxis_tickangle=45,
    xaxis_title='',
    legend = dict(x = .75, y = .94),
    shapes=[
        dict(
            type='line',
            yref='y2', y0=2, y1=2,
            xref='paper', x0=0, x1=.94,
            line=dict(
                color="red",
                width=2,
                dash="dot",
            )
        )],
    annotations=[
        dict(
            x=results.iloc[-2, 0],
            y= 3,
            xref="x",
            yref="y2",
            text="|t|=2",
            font=dict(
                color="red",
                size=14
            ),
            showarrow=False
            )]
)
fig.update_yaxes(title='', tickformat='.0%', secondary_y=False)
fig.update_yaxes(title='|t stat|', tickformat='.0f', showgrid = False, secondary_y=True)
fig.show()