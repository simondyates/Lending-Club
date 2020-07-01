import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def score_regression(model, X_test, Y_test, prints=True, save_results=True):
    model_name = type(model).__name__
    Y_test_hat = model.predict(X_test)

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

    dt_stamp = datetime.now().strftime('%m-%d_%H-%M')

    data = [model_name, bias, maxDev, meanAbsDev, MSE ** 0.5, R2, Adj_R2, skew, kurt, AIC, F]
    idx = ['Model', 'Bias', 'MaxDev', 'MeanAbsDev', 'RMSE', 'R2', 'Adj_R2', 'Skew', 'Kurt', 'AIC', 'F']
    results = pd.Series(data, index=idx)

    if (prints):
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
        results.to_csv(f'./Model_results/{model_name[:10]}_{dt_stamp}.csv')


def plot_roc(Y_train_hat, Y_test_hat, Y_train, Y_test, model_name):
    AUC_train = roc_auc_score(Y_train, Y_train_hat)
    AUC_test = roc_auc_score(Y_test, Y_test_hat)

    fp_train, tp_train, _ = roc_curve(Y_train, Y_train_hat)
    fp_test, tp_test, thresholds = roc_curve(Y_test, Y_test_hat)

    precision = 1/(1+ (1/np.mean(Y_test)-1)*(fp_test/tp_test))
    thresholds[0]=1  #sklearn may produce arbitrary first threshold > 1

    ref_thresholds = np.linspace(min(thresholds), max(thresholds),1000)
    selected = [np.nanmean(Y_test_hat>p) for p in ref_thresholds]

    _,axes = plt.subplots(1, 2, figsize=(14, 6))
    ax0,ax1=axes[0],axes[1]

    ax0.plot(fp_train, tp_train, label=f'Train. AUC={AUC_train:.4f}')
    ax0.plot(fp_test, tp_test, label=f'Test. AUC={AUC_test:.4f}')
    ax0.plot((0,1),(0,1), color='grey')
    ax0.plot((np.mean(Y_train)), (np.mean(Y_train)), color='grey', marker='o', label='random')
    ax0.set_xlabel('false positive rate')
    ax0.set_ylabel('recall')
    ax0.set_title(f'ROC: {model_name}')
    ax0.legend(loc='lower right')
    ax0.grid(True)

    ax1.plot(thresholds, precision, label=f'precision')
    ax1.plot(ref_thresholds, selected, label=f'selected %')
    ax1.set_xlabel('cutoff thresholds')
    ax1.set_title(f'Precision and selected %')
    ax1.legend(loc='lower right')
    ax1.invert_xaxis()
    ax1.grid(True)

    plt.show()


def plot_report_metrics(df_report, test_mode):
    fig,axes = plt.subplots(1, 2, figsize=(14, 5))
    ax0,ax1=axes[0],axes[1]
    model_name = df_report.model[0]

    for col in ['prec_5','prec_10','prec_20','support']:
        ax0.plot(df_report.run, df_report[col]/df_report['support'], label=col)
    ax0.set_title(f'{test_mode} OOS   {model_name}   Lift')
    ax0.set_ylabel('lift')
    ax0.legend()

    for col in ['support','AUC_test']:
        ax1.plot(df_report.run, df_report[col], label=col)
    ax1.set_title(f'{test_mode} OOS   {model_name}   AUC')
    ax1.legend()

    plt.show()
