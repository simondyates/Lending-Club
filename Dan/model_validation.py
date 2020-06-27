import numpy as np
import pandas as pd
from datetime import datetime

def score_regression(model, X_test, Y_test, print=True, save_results=True):
    model_name = type(model).__name__
    Y_test_hat = model_name.predict(X_test)

    n = len(Y_test_hat)
    p = X_test.shape[1]

    bias = (Y_test_hat - Y_test).mean()
    maxDev = abs(Y_test - Y_test_hat).max()
    meanAbsDev = abs(Y_test - Y_test_hat).mean()
    MSE = ((Y_test_hat - Y_test) ** 2).mean()
    MTE = ((Y_test - Y_test.mean()) ** 2).mean()
    MSM = ((Y_test_hat - Y_test.mean()) ** 2).mean()
    R2 = 1 - MSE / MTE
    Adj_R2 = 1 - ((1 - R2) * (n - 1) / (n - p - 1))
    skew = ((Y_test_hat - Y_test) ** 3).mean() / MSE ** 1.5
    kurt = ((Y_test_hat - Y_test) ** 4).mean() / MSE ** 2 - 3
    AIC = n * np.log(MSE) + 2 * (p + 1)
    F = (MSM / MSE) * ((p - 1) / (n - p))

    dt_stamp = datetime.now().strftime('%m-%d %H-%M')

    data = [model_name, bias, maxDev, meanAbsDev, MSE ** 0.5, R2, Adj_R2, skew, kurt, AIC, F, dt_stamp]
    idx = ['Model', 'Bias', 'MaxDev', 'MeanAbsDev', 'RMSE', 'R2', 'Adj_R2', 'Skew', 'Kurt', 'AIC', 'F']
    results = pd.Series(data, index=idx)

    if (print):
        print('-' * len(model_name))
        print(model_name)
        print('-' * len(model_name))
        print(f'Bias: {bias:,.0f}')
        print(f'Max Dev: {maxDev:,.0f}')
        print(f'Mean Abs Dev: {meanAbsDev:,.0f}')
        print(f'RMSE: {MSE ** 0.5:,.0f}')
        print(f'R^2: {R2:.2%}')
        print(f'Adj R^2: {Adj_R2:.2%}')
        print(f'Resid skew: {skew:.2f}')
        print(f'Resid kurt: {kurt:.2f}')
        print(f'AIC: {AIC:,.2f}')
        print(f'F: {F:.2f}')
        print('-' * len(model_name))
    if (save_results):
        results.to_csv(f'./Model_results/{model_name[:10]} {dt_stamp}.csv')