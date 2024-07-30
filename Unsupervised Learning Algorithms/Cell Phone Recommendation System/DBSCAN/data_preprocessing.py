from sklearn.preprocessing import StandardScaler, LabelEncoder 

def preprocess_data(data):
    label_encoders = {}
    for column in ['brand', 'model', 'operating system']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le 
    
    scaler = StandardScaler()
    numeric_features = ['internal memory', 'RAM', 'performance', 
                        'main camera', 'selfie camera', 'battery size',
                        'screen size', 'weight', 'price']
        
    data[numeric_features] = scaler.fit_transform(data[numeric_features])

    return data, label_encoders, scaler 
