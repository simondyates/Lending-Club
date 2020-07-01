import pandas as pd

from sklearn.model_selection import train_test_split

from model_validation import plot_roc, plot_report_metrics
from train_model import tune_model
from helpers import normalize_arr

mode = 'recent'

df = pd.read_pickle(f'./derivedData/df_train_{mode}_wReturns.pkl')

# ------------------------- Features & Targets ---------------------------

ordinal = [
    'application_type',
    'grade',
    'sub_grade',
    'initial_list_status',  # LC claims this is purely random: chi2 test!
    'emp_length',
    'verification_status'
]

nominal = [
  'purpose',
  'addr_state', # if linear models do not pick up this feature, revisit amd remove dummies (probably too much noise)
  'home_ownership',
  'disbursement_method', #highly imbalanced and probably irrelevant
]

numeric = [
    'loan_amnt', # numeric
    'int_rate',  # numeric
    'installment',  # numeric
    'annual_inc', # numeric
    'fico_range_low',
    'delinq_2yrs',# numeric
    'dti', # numeric
    'open_acc', # numeric
    'pub_rec', # numeric
    'revol_bal', # numeric
    'revol_util', # numeric
    '_credit_hist', # new numeric
    'term'
]

target = '_good'

X = df[nominal+ordinal+numeric]
Y = df[target]

# ---------------------------- Preprocessing ----------------------------

sub_grades = sorted(df.sub_grade.unique())
sub_grades_dict = {x:sub_grades.index(x)+1 for x in sub_grades}  # map 'A1' to 1 rather than 0
sub_grades_dict_reverse = {v:k for k,v in sub_grades_dict.items()} # for visual purposes, need later

emp_length_dict =  {'< 1 year':0, '1 year':1, '2 years':2, '3 years':3, '4 years':4, '5 years':5,
                    '6 years':6, '7 years':7, '8 years':8, '9 years':9, '10+ years':10,}
ordinal_dict = {
    'application_type': {'Individual':0, 'Joint App':1},
    'grade':   {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7},
    'sub_grade': sub_grades_dict,
    'initial_list_status': {'w':0, 'f':1},
    'emp_length': emp_length_dict,
    'verification_status': {'Not Verified':0, 'Source Verified':1, 'Verified':2},  #'source verified' is a softer check
    '_has_desc': {False:0, True:1}
}

for f in ordinal:
    X[f].replace(ordinal_dict[f], inplace=True)

normalize = True
dummify = True

if dummify:
    X = pd.get_dummies(X, columns=nominal)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=42)

if dummify & normalize:
    X_train, X_test = normalize_arr(X_train, X_test)

# ------------------------ Classification ------------------------------

list_models = ['logistic']

grids = tune_model(list_models, X_train, Y_train)

Y_train_hat = grids['logistic'].best_estimator_.predict(X_train)
Y_test_hat = grids['logistic'].best_estimator_.predict(X_test)

plot_roc(Y_train_hat, Y_test_hat, Y_train, Y_test, 'logistic')