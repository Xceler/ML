import pandas as pd 

def recommend_cellphones(user_id, data, kmeans, top_n = 5):
    user_ratings = data[data['user_id'] == user_id]
    if user_ratings.empty:
        return pd.DataFrame()
    
    user_clusters = user_ratings['cluster'].unique()
    recommend_cellphones = data[data['cluster'].isin(user_clusters)]
    recommned_cellphones = recommend_cellphones[~recommend_cellphones['cellphone_id'].isin(user_ratings['cellphone_id'])]

    top_recommendations = recommend_cellphones.groupby('cellphone_id').agg({'rating' : 'mean'}).sort_values(by= 'rating', ascending = False).head(top_n)

    return top_recommendations 
