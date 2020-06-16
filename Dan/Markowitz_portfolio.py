import numpy as np
from scipy.optimize import minimize

def Markowitz_portfolio(ret_def, ret_notdef, prob_def, loan_amount, inv_amount, risk_tol):
    # risk_returns = ExpectedShortfall(k-means)
    risk_returns = 'tbd'
    n = len(ret_def)

    returns = prob_def*ret_def + (1 - prob_def)*ret_notdef
    objective = lambda w: np.transpose(w).dot(risk_returns).dot(w) - risk_tol*(np.transpose(returns).dot(w))

    # Can't invest more than what you have
    cst1 = lambda w: 1-sum(w)
    cst2 = lambda w: sum(w)-.9
    # Can't invest less than 25$ on a single loan
    # cst2 = lambda w: 25-w*inv_amount

    # Can't invest more than the loan amount
    cst3 = lambda w: loan_amount-w*inv_amount

    constraints = (
        {'type': 'ineq', 'fun': cst1},
        {'type': 'ineq', 'fun': cst2},
        {'type': 'ineq', 'fun': cst3}
    )

    x0 = np.zeros(n)

    res = minimize(objective, x0=x0, method='SLSQP', constraints=constraints)

    exp_port = res['x'].dot(ret_def)
    var_port = np.transpose(res['x']).dot(covarMat).dot(res['x'])

    print(f'Expected return of the portfolio : {exp_port}')
    print(f'Variance of the portfolio : {var_port}')

    print(res)


muVec = [.1, .09, .08, .07, .3]
covarMat = [[.16, 0, 0, 0, 0],
            [0, .09, 0, 0, 0],
            [0, 0, 0.06, 0, 0],
            [0, 0, 0, .07, 0],
            [0, 0, 0, 0, .21]]
capVec = [1000, 1000, 1000, 1000, 1000]
investAmount = 150
risk_tol = .5
Markowitz_portfolio(muVec, covarMat, capVec, investAmount, risk_tol)