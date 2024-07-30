from sklearn.cluster import DBSCAN 

def cluster_data(data, features, eps = 0.5, min_samples = 5):
    x = data[features]
    dbscan = DBSCAN(eps = eps, min_samples = min_samples)
    data['cluster'] = dbscan.fit_predict(x)
    return dbscan 

    