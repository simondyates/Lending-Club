import numpy as np
from scipy.optimize import minimize

def Markowitz_portfolio(exp_returns, covar_returns, loan_amount, inv_amount, risk_tol):
    exp_returns = np.array(exp_returns)
    covar_returns = np.array(covar_returns)
    loan_amount = np.array(loan_amount)
    n = len(exp_returns)

    objective = lambda w: np.transpose(w).dot(covar_returns).dot(w) - risk_tol*(np.transpose(exp_returns).dot(w))

    # Can't invest more than what you have
    cst1 = lambda w: 1-sum(w)

    # Can't invest less than 25$ on a single loan
    cst2 = lambda w: 25-w*inv_amount

    # Can't invest more than the loan amount
    cst3 = lambda w: loan_amount-

    constraints = (
        {'type': 'ineq', 'fun': cst1},
        {'type': 'ineq', 'fun': cst2},
        {'type': 'ineq', 'fun': cst3}
    )

    x0 = np.zeros(n)

    res = minimize(objective, x0=x0, method='SLSQP', constraints=constraints)

    exp_port = (res.x*inv_amount).dot(exp_returns)
    var_port =

    print(f'Expected return of the portfolio : {exp_port}')
    print(f'Variance of the portfolio : {var_port}')

    print(res)


muVec = [.1, .09, .08, .07]
covarMat = [[.16, 0, 0, 0],
            [0, .09, 0, 0],
            [0, 0, .08, 0],
            [0, 0, 0, .07]]
capVec = [1000, 1000, 1000, 1000]
investAmount = 100
risk_tol = .5
Markowitz_portfolio(muVec, covarMat, capVec, investAmount, risk_tol)