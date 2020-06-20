# Implements Gradient Boosting regression on loan_status
# Currently this underperforms a univariate regression using sub_grade alone

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from joblib import dump
import matplotlib as mpl

# Global Variables
scale = True

accept = pd.read_pickle('../derivedData/train.pkl')

# Split target from attributes and normalise attribs
y = (accept['PV'] / accept['funded_amnt']).to_numpy()
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
    X_train_s = ct.fit_transform(X_train)
    X_test_s = ct.transform(X_test)

# Run gradient boost regressor
boost = GradientBoostingRegressor(subsample=0.1, verbose=1)
boost.fit(X_train_s, y_train)
print(f'IS R^2: {boost.score(X_train_s, y_train):.2%}')
print(f'OOS R^2: {boost.score(X_test_s, y_test):.2%}')

# Now, invest $100,000 in the 100 best loans in test
# Current code assumes each loan is at least $1k size (I think this is true though)
pvs = pd.Series(boost.predict(X_test_s)).sort_values()
selected = pvs.index[-100:]
pv_sel = sum(y_test[selected]/100)
rand = np.random.choice(pvs.index, 100, replace=False)
pv_rand = sum(y_test[rand])/100
print(f'PV of selected: {pv_sel:,.1%}')
print(f'PV of random: {pv_rand:,.1%}')

# Save the model for later use
#dump(boost, '../derivedData/boost.joblib')
#dump(ct, '../derivedData/boost_scaler.joblib')

# Plot the distribution of predictions
ax2 = pvs.hist(bins=100, label='pred')
pd.Series(y_test).hist(ax=ax2, bins=100, label='actual', alpha=.2)
ax2.legend()
ax2.get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0%}'))
ax2.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax2.set_title('Actual vs Predicted Returns')
