# Implements Gradient Boosting regression on loan_status
# Currently this underperforms a univariate regression using sub_grade alone

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

# Global Variables
scale = True

accept = pd.read_pickle('../derivedData/train.pkl')
accept.set_index('id', inplace = True)

# Split target from attributes and normalise attribs
y = accept['loan_status']
X = accept.drop('loan_status', axis=1)

# Drop attributes with updates after loan inception
leaks = ['recoveries', 'total_pymnt', 'dti', 'last_pymnt_d',
         'revol_util', 'open_acc', 'pub_rec', 'revol_bal',
         'revol_util', 'delinq_2yrs']
X = X.drop(leaks, axis=1)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale (if desired)
if scale:
    date_cols = ['issue_d','earliest_cr_line']
    num_cols = ['loan_amnt', 'funded_amnt', 'term', 'int_rate', 'grade',
       'sub_grade', 'emp_length', 'home_ownership', 'annual_inc',
       'fico_range_high', 'fico_range_low']
    cols_to_scale = date_cols + num_cols
    ct = make_column_transformer((StandardScaler(), cols_to_scale), remainder='passthrough')
    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)

# Run gradient boost classifier
boost = GradientBoostingClassifier(subsample=0.1, verbose=1)
boost.fit(X_train, y_train)
fpr, tpr, _ = metrics.roc_curve(y_test, boost.predict_proba(X_test)[:, 0], pos_label=0)
auc = metrics.auc(fpr, tpr)
print(f'AUC: {auc:.2%}')
print(f'R^2: {boost.score(X_test, y_test):.2%}')