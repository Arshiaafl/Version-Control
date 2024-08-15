import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    if abs(class_corr[i]) > 0.5:
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

model = Sequential()

model.add(Dense(60, input_shape= (X_test.shape[1],), activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)


mlflow.set_experiment("Fraud Detection")

with mlflow.start_run(run_name="Keras NN"):
    mlflow.log_param("model", "Keras NN")
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("num_epochs", 10)
    mlflow.log_metric('accuracy', history.history['accuracy'][-1])
    mlflow.log_metric('val_accuracy', history.history['val_accuracy'][-1])
    mlflow.keras.log_model(model, "model")