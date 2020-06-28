import numpy as np
import time
import pickle
from datetime import datetime

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


def tune_model(models, X, Y, save_results=True):
    verbose = 1

    models = [x.lower() for x in models]

    grid_search = {}

    # ----------------------------- Ridge ------------------------------
    if 'ridge' in models:
        print(f'===== Training Ridge =====')
        tic = time.time()

        ridge = Ridge()

        params = {
            'alpha': np.linspace(1e-3, 50, 10)
        }

        grid_search['ridge'] = GridSearchCV(ridge, params, scoring='r2', cv=6, verbose=verbose, n_jobs=4)

        grid_search['ridge'].fit(X, Y)

        print(f'Time to tune : {time.time() - tic} sec')

        if save_results:
            dt_stamp = datetime.now().strftime('%m-%d_%H-%M')
            filename = f'./Model_results/ridge_{Y.name}_{dt_stamp}.sav'
            pickle.dump(grid_search['ridge'], open(filename, 'wb'))
    # ----------------------------- Lasso -------------------------------
    if 'lasso' in models:
        print(f'===== Training Lasso =====')
        tic = time.time()

        lasso = Lasso()

        params = {
            'alpha': np.linspace(1e-3, 50, 10),
            'selection': ['cyclic', 'random']
        }

        grid_search['lasso'] = GridSearchCV(lasso, params, scoring='r2', cv=6, verbose=verbose, n_jobs=4)

        grid_search['lasso'].fit(X, Y)

        print(f'Time to tune : {time.time() - tic} sec')

        if save_results:
            dt_stamp = datetime.now().strftime('%m-%d_%H-%M')
            filename = f'./Model_results/lasso_{Y.name}_{dt_stamp}.sav'
            pickle.dump(grid_search['lasso'], open(filename, 'wb'))
    # ----------------------------- Random Forest ---------------------------
    if 'randomforest' in models:
        print('===== Random Forest =====')
        tic = time.time()

        rf = RandomForestRegressor()

        params = {
            'n_estimators':[100, 250, 500],
            'max_depth':[2,3,4],
            'min_samples_leaf':[100, 1000, 5000],
            'n_jobs':[6]
        }

        grid_search['random_forest'] = GridSearchCV(rf, params, scoring='r2', cv=6, verbose=3)

        grid_search['random_forest'].fit(X, Y)

        print(f'Time to tune : {np.round((time.time() - tic)/60)} min')

        if save_results:
            dt_stamp = datetime.now().strftime('%m-%d_%H-%M')
            filename = f'./Model_results/RF_{Y.name}_{dt_stamp}.sav'
            pickle.dump(grid_search['random_forest'], open(filename, 'wb'))

    return grid_search
