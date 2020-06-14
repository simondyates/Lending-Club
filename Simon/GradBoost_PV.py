# Implements Gradient Boosting regression on loan_status
# Currently this underperforms a univariate regression using sub_grade alone

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Global Variables
scale = True

accept = pd.read_pickle('../derivedData/train.pkl')
#accept.set_index('id', inplace = True)

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

# Run gradient boost regressor
boost = GradientBoostingRegressor(subsample=0.1, verbose=1)
boost.fit(X_train, y_train)
print(f'IS R^2: {boost.score(X_train, y_train):.2%}')
print(f'OOS R^2: {boost.score(X_test, y_test):.2%}')

# Now, invest $100,000 in the 100 best loans in test
# Current code assumes each loan is at least $1k size (I think this is true though)
pvs = pd.Series(boost.predict(X_test)).sort_values()
selected = pvs.index[-100:]
pv_sel = y_test[selected].sum()
rand = np.random.choice(pvs.index, 100, replace=False)
pv_rand = y_test[rand].sum()
print(f'PV of selected: {pv_sel:,.0f}')
print(f'PV of random: {pv_rand:,.0f}')

