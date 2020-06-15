# Simulate a trading strategy as follows:
# Review each loan chronologially and invest if predicted PV > loan_amnt * (1 + min_ret)
# Continue until the full amount ($100,000) has been invested
# Then, purchase new loans meeting PV criteria only to reinvest cashflows received

import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split

# Tactic parameters
max_per_loan = 1000
max_portfolio = 100000
min_ret = .1

accept = pd.read_pickle('../derivedData/train.pkl')
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
    def __init__(self, universe):
        self.__univ = universe
        self.ids = []
        self.amnts = []
        self.spent = 0
        self.received = 0

    def add(self, id, amnt):
        self.ids.append(id)
        self.amnts.append(amnt)
        self.spent += amnt

    def PV(self):
        return(self.__univ.loc[self.ids, 'PV'].sum())

boost = load('../derivedData/boost.joblib')
port = Portfolio(accept)

i = 0
while port.spent < max_portfolio:
    pv = boost.predict(X_train_s[i, :].reshape(1, -1))
    amnt = X_train.iloc[i, 1]
    id = X_train.index[i]
    if (pv/amnt >= min_ret):
        port.add(id, max_per_loan)
    i = i+1

print(f'Total invested: {port.spent:,.0f}')
print(f'Total returned: {port.PV():,.0f}')

# Next: do each month as a batch
# Figure out how much cash is coming back that month
# Purchase approriately