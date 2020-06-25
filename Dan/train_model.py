import numpy as np
import time

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

def train_model(models, X, Y):
    verbose = 1

    models = [x.lower() for x in models]

    tic = time.time()

    grid_search = {}

    # ----------------------------- Ridge ------------------------------
    if 'ridge' in models:
        print(f'===== Training Ridge =====')
        ridge = Ridge()

        params = {
            'alpha': np.linspace(1e-3, 50, 10)
        }

        grid_search['ridge'] = GridSearchCV(ridge, params, scoring='r2', cv=6, verbose=verbose, n_jobs=4)

        grid_search['ridge'].fit(X, Y)

        print(f'Time to train : {time.time() - tic} sec')

    # ----------------------------- Lasso -------------------------------
    if 'lasso' in models:
        print(f'===== Training Lasso =====')
        lasso = Lasso()

        params = {
            'alpha': np.linspace(1e-3, 50, 10),
            'selection': ['cyclic', 'random']
        }

        grid_search['lasso'] = GridSearchCV(lasso, params, scoring='r2', cv=10, verbose=verbose, n_jobs=4)

        grid_search['lasso'].fit(X, Y)

        print(f'Time to train : {time.time() - tic} sec')

    return grid_search
