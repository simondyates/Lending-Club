import numpy as np
import pandas as pd
from scipy.optimize import minimize
from gurobipy import *

def Markowitz_portfolio(return_loan, loan_amount, inv_amount, risk_tol):
    '''
    :param return_loan: list of returns predicted, if default and if non default
    :param loan_amount: maximum investment for each loan
    :param inv_amount: money amount invested in the portfolio
    :param risk_tol: risk tolerance of the investor
    :return: list of weights of the portfolio, returns
    '''

    # K = 3
    # risk_returns = Risk_metric(K)

    n = len(returns_loan)

    objective = lambda w: np.transpose(w).dot(w) - risk_tol*(np.transpose(return_loan).dot(w))

    # Can't invest more than what you have
    cst1 = lambda w: 1-sum(w)
    cst2 = lambda w: sum(w)-.9

    # Can't invest more than the loan amount
    cst3 = lambda w: loan_amount-np.repeat(w*inv_amount, len(loan_amount))

    # Risk tolerance constraint
    cst4 = lambda w: risk_tol - np.transpose(w).dot(risk_returns)

    constraints = (
        {'type': 'ineq', 'fun': cst1},
        {'type': 'ineq', 'fun': cst2},
        {'type': 'ineq', 'fun': cst3},
        {'type': 'ineq', 'fun': cst4}
    )

    x0 = np.zeros(n)

    res = minimize(objective, x0=x0, method='SLSQP', constraints=constraints)

    # exp_port = res['x'].dot(returns_loan)
    # var_port = np.transpose(res['x']).dot(covarMat).dot(res['x'])

    # print(f'Expected return of the portfolio : {exp_port}')
    # print(f'Variance of the portfolio : {var_port}')

    # print(res)


ret_def = pd.read_csv('./rawData/predicted_returns_defaulting.csv')
ret_notdef = pd.read_csv('./rawData/predicted_returns_non_defaulting.csv')

# Temporary
returns_loan = pd.concat([ret_def, ret_notdef])
maxloan = np.repeat(10000, len(returns_loan))
invest = 50000
risk_tol = .4

Markowitz_portfolio(returns_loan['annualized_ret'], maxloan, invest, risk_tol)

# prob_def = np.random.rand(len(returns_loan))
#
# muVec = [.1, .09, .08, .07, .3]
# covarMat = [[.16, 0, 0, 0, 0],
#             [0, .09, 0, 0, 0],
#             [0, 0, 0.06, 0, 0],
#             [0, 0, 0, .07, 0],
#             [0, 0, 0, 0, .21]]
# capVec = [1000, 1000, 1000, 1000, 1000]
# investAmount = 150
# risk_tol = .5
# Markowitz_portfolio(muVec, covarMat, capVec, investAmount, risk_tol)