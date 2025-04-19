import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from GenreClassificationsKnn import KNN


# Load features_3_sec.csv
data = pd.read_csv('features_3_sec.csv')

X = data.drop(columns=["filename", "length", "label"]) 
Y = data["label"]

encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X.values, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNN()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

accuracy = np.mean(predictions == y_test)
print(f"Accuracy:", accuracy)