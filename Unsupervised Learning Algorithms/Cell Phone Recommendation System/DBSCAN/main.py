from data_loader import load_data, merge_data 
from data_preprocessing import preprocess_data 
from clustering import cluster_data 
from recommendation import recommend_cellphones 
from data_updater import update_data 

def main():
    ratings, cellphones, users = load_data()
    data = merge_data(ratings, cellphones, users)
    data, label_encoders, scaler = preprocess_data(data)

    numeric_features = ['internal memory', 'RAM', 'performance',
                        'main camera', 'selfie camera', 'battery size',
                        'screen size', 'weight', 'price']
        
    features = numeric_features + ['brand', 'model', 'operating system']

    dbscan = cluster_data(data, features)

    user_id = 27 
    recommendations = recommend_cellphones(user_id, data, dbscan)
    print("Recommendations for User ID:", user_id)
    print(recommendations)

    data = update_data('Unsupervised Learning Algorithms/Cell Phone Recommendation System/Dataset/cellphones ratings.csv',
                       'Unsupervised Learning Algorithms/Cell Phone Recommendation System/Dataset/cellphones data.csv',
                       'Unsupervised Learning Algorithms/Cell Phone Recommendation System/Dataset/cellphones users.csv',
                       data, features, dbscan)
    
    print('Updated Data:')
    print(data.head())


if __name__ == '__main__':
    main()