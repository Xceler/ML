from sklearn.cluster import MiniBatchKMeans 

def cluster_data(data, features, n_clusters = 10):
    x = data[features]
    kmeans = MiniBatchKMeans(n_clusters = n_clusters, random_state = 0)
    data['cluster'] = kmeans.fit_predict(x)
    return kmeans 

    