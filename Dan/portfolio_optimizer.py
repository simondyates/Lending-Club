import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

def portfolio_optimizer(returns_df, funded_amount, budget, size, risk_tol):
    '''
    :param returns:
    :param funded_amount:
    :param budget:
    :param size:
    :param risk_tol:
    :return:
    '''

    returns = returns_df.annualized_ret

    n = len(returns)

    objective = lambda x: -sum(x[0][i] * returns[i] * x[1][i] for i in range(n))

    cst0 = lambda x: x[0][:] * (x[0][:] - 1) # Either we invest in a loan, either not
    cst1 = lambda x: size - sum(x[0][:]) # We invest in less loans than the size of the portfolio
    cst2 = lambda x: sum(x[0][:]) - int(.9 * size) # We invest in more than 90% of the size of the portfolio
    cst3 = lambda x: budget - sum(x[0][:] * x[1][:]) # We can't invest more than our budget

    # Can't invest more than the loan founded amount
    var_bounds = [(0, funded_amount[i]) for i in range(len(funded_amount))]

    # Risk tolerance constraint
    # cst5 = lambda: 1

    constraints = (
        {'type': 'eq', 'fun': cst0},
        {'type': 'ineq', 'fun': cst1},
        {'type': 'ineq', 'fun': cst2},
        {'type': 'ineq', 'fun': cst3}
    )

    x0 = np.zeros(n)
    a0 = np.zeros(n)

    res = minimize(objective,
                   x0=[x0, a0],
                   method='SLSQP',
                   constraints=constraints,
                   bounds=var_bounds)

    return res


# Test portfolio
ret_def = pd.read_csv('./rawData/predicted_returns_defaulting.csv')
ret_notdef = pd.read_csv('./rawData/predicted_returns_non_defaulting.csv')

returns_loan = pd.concat([ret_def, ret_notdef])
founded_amt = np.repeat(10000, len(returns_loan))
budget = 1000000
size = 1000
risk_tol = .4

portfolio_optimizer(returns_loan, founded_amt, budget, size, risk_tol)