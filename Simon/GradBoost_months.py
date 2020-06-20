# Implements Gradient Boosting regression on loan_status
# Currently this underperforms a univariate regression using sub_grade alone

import pandas as pd
import datetime as dt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from joblib import dump
import matplotlib

# Global Variables
scale = True

accept = pd.read_pickle('../derivedData/train.pkl')
accept.set_index('id', inplace=True)

# Create target
def to_dt(i):
    return(dt.datetime.strptime(str(i), '%Y%m%d'))
last_dt = accept['last_pymnt_d'].map(to_dt)
first_dt = accept['issue_d'].map(to_dt)
y = (last_dt - first_dt).map(lambda x: x.days)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(accept, y, test_size=0.2, random_state=42)

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

# Remove informative columns
leaks = ['PV', 'loan_status', 'last_pymnt_d',
         'recoveries', 'total_pymnt', 'dti',
         'revol_util', 'open_acc', 'pub_rec', 'revol_bal',
         'revol_util', 'delinq_2yrs']
drp = [X_train.columns.get_loc(c) for c in leaks]
keep = [i for i in range(len(X_train.columns)) if i not in drp]
X_train_s = X_train_s[:, keep]
X_test_s = X_test_s[:, keep]

# Run gradient boost regressor
boost = GradientBoostingRegressor(subsample=0.1, verbose=1)
boost.fit(X_train_s, y_train)
print(f'IS R^2: {boost.score(X_train_s, y_train):.2%}')
print(f'OOS R^2: {boost.score(X_test_s, y_test):.2%}')

# Now, invest $100,000 in the 100 best loans in test
# Current code assumes each loan is at least $1k size (I think this is true though)
dts = pd.Series(boost.predict(X_test_s)).sort_values()
selected = dts.index[-100:]
pv_col = X_test.columns.get_loc('PV')
amnt_col = X_test.columns.get_loc('funded_amnt')
pv_sel = X_test.iloc[selected, pv_col] @ (1000/X_test.iloc[selected, amnt_col])
rand = np.random.choice(dts.index, 100, replace=False)
pv_rand = X_test.iloc[rand, pv_col] @ (1000/X_test.iloc[rand, amnt_col])
print(f'PV of selected: {pv_sel:,.0f}')
print(f'PV of random: {pv_rand:,.0f}')

# Save the model for later use
#dump(boost, '../derivedData/boost.joblib')
#dump(ct, '../derivedData/boost_scaler.joblib')

# Plot the distribution of predictions
ax1 = dts.hist(bins=100, label='pred')
pd.Series(y_test).hist(ax=ax1, bins=100, label='actual', alpha=.2)
ax1.legend()
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax1.set_title('Actual vs Predicted Lives')
