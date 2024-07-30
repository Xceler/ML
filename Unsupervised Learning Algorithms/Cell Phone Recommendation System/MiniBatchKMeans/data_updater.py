import pandas as pd 


def update_data(new_rating, new_cellphone, new_user, data, features, kmeans):
    new_ratings = pd.read_csv(new_rating)
    new_cellphones = pd.read_csv(new_cellphone)
    new_users = pd.read_csv(new_user)

    new_data = pd.merge(new_ratings, new_cellphones, on = 'cellphone_id')
    new_data = pd.merge(new_data, new_users, on = 'user_id')

    data = pd.concat([data, new_data], ignore_index = True)

    new_data[features] = data[features]
    new_clusters = kmeans.predict(new_data[features])
    data.loc[new_data.index, 'cluster'] = new_clusters 
    return data 