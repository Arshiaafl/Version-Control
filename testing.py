import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import mlflow



df = pd.read_csv('creditcard_2023.csv', index_col="id")
#print(df.head())

matrix = df.corr()
#print(matrix)

class_corr = matrix.iloc[:, -1]

selected_features = []


for i in range(len(class_corr)):
    if abs(class_corr[i]) > 0.4:
        # Append the feature name to the list
        selected_features.append(class_corr.index[i])


#print("Selected features with correlation > 0.5 with the class:")
#print(selected_features)

final_df = df[selected_features]
#print(final_df.head())

Y = final_df['Class']
X = final_df.drop(['Class'], axis=1)

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, shuffle = True)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import mlflow.pyfunc

# Load the latest model from production
model_name = "Keras NN"
model_version = "latest"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

# Use the model for predictions
predictions = model.predict(X_test)

accuracy_score(y_test, predictions.round(), normalize=False)
print(accuracy_score)
