from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pandas as pd
import numpy as np


def normalize_arr(fit_arr, *arrs):
    sc = StandardScaler()
    fit_arr = sc.fit_transform(fit_arr)
    out=[]
    for arr in arrs:
        out.append(sc.transform(arr))
    return [fit_arr]+out


def save_predictions(Y_test, Y_test_hat, test_mode, path):
    timestamp = datetime.now().strftime('%m-%d_%H-%M')
    filename = f'predictions_{test_mode}_{timestamp}.csv'

    Y_test_hat = pd.Series(np.squeeze(Y_test_hat), name='p', index=Y_test.index)
    predictions = pd.concat([Y_test, Y_test_hat], axis=1)
    predictions.to_csv(path + 'features/' + filename)

    print(f'saved: {filename}')


def save_report(df_report, test_mode, path):
    timestamp = datetime.now().strftime('%m-%d_%H-%M')
    filename = f'report_{test_mode}_{timestamp}.csv'

    df_report.to_csv(path + 'reports/' + filename)

    print(f'saved: {filename}')

