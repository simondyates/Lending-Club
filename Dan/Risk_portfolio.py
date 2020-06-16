from sklearn.cluster import KMeans
import numpy as np

def Expected_shortfall():
    pass

def Risk_metric(data, K = 3, type='ES'):
    '''

    :param data: Data file
    :param K: Number of clusters, eg, number of risk categories
    :param type: Type of risk metrics used (ES : expected shortfall, std : standard deviation)
    :return: list of risks values
    '''

    km = KMeans(init=data.iloc[np.random.choice(K)], n_clusters=K, max_iter=1000)

    km.fit(data)
    clusters_pred = km.predit(data)

    data['cluster'] = clusters_pred

    if type == 'ES':
        for k in np.unique(clusters_pred):
            risk = Expected_shortfall(data[data['cluster'] == k])

    return risk