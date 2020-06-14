#  Implements univariate logistic regression on loan_status

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

# Global Variables
scale = True

accept = pd.read_pickle('../derivedData/train.pkl')
accept.set_index('id', inplace = True)

# Split target from attributes and normalise attribs
Y = accept['loan_status']
X = accept.drop('loan_status', axis=1)

# Drop attributes with updates after loan inception
leaks = ['recoveries', 'total_pymnt', 'dti', 'last_pymnt_d',
         'revol_util', 'open_acc', 'pub_rec', 'revol_bal',
         'revol_util', 'delinq_2yrs']
X = X.drop(leaks, axis=1)

if scale:
    date_cols = ['issue_d','earliest_cr_line']
    num_cols = ['loan_amnt', 'funded_amnt', 'term', 'int_rate', 'grade',
       'sub_grade', 'emp_length', 'home_ownership', 'annual_inc',
       'fico_range_high', 'fico_range_low']
    cols_to_scale = date_cols + num_cols
    ct = make_column_transformer((StandardScaler(), cols_to_scale), remainder='passthrough')
    ct.fit_transform(X)

results = pd.DataFrame(index=X.columns, columns=['Beta', 'R2', 'AUC'])

# Run univariate logit regression on each feature
logr = LogisticRegression(class_weight='balanced')

for col in X.columns:
    print(col)
    logr.fit(X[[col]], Y)
    fpr, tpr, _ = metrics.roc_curve(Y, logr.predict_proba(X[[col]])[:, 0], pos_label=0)
    auc = metrics.auc(fpr, tpr)
    results.loc[col] = [logr.coef_[0, 0], logr.score(X[[col]], Y), auc]

results = results.sort_values('AUC', ascending = False)
top20 = results.iloc[0:20]

# Plot the results
fig1=go.Figure()
fig1.add_trace(go.Bar(x=top20.index, y=top20['AUC'], name='AUC'))
fig1.update_layout(
     title='Univariate AUC of Top 20 Features',
     xaxis_title='',
     yaxis_title='AUC',
     yaxis_tickformat='.0%',
     shapes=[
         dict(
             type='line',
             yref='y', y0=.5, y1=.5,
             xref='paper', x0=0, x1=1,
             line=dict(
                 color="red",
                 width=2,
                 dash="dot",
              )
          )]
)
fig1.show()