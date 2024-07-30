import pandas as pd 

def load_data():
    ratings = pd.read_csv('Unsupervised Learning Algorithms/Cell Phone Recommendation System/Dataset/cellphones ratings.csv')
    cellphones = pd.read_csv('Unsupervised Learning Algorithms/Cell Phone Recommendation System/Dataset/cellphones data.csv')
    users = pd.read_csv('Unsupervised Learning Algorithms/Cell Phone Recommendation System/Dataset/cellphones users.csv')
    return ratings, cellphones, users 


def merge_data(ratings, cellphones, users):
    data = pd.merge(ratings, cellphones, on ='cellphone_id')
    data = pd.merge(data, users, on = 'user_id')
    return data 

