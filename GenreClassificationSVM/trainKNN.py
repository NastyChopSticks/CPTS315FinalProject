import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from GenreClassificationsKnn import KNN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def LoadTrain(csv_path):
    # Load features_3_sec.csv
    data = pd.read_csv(csv_path)

    X = data.drop(columns=["filename", "length", "label"]) 
    Y = data["label"]

    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)

    # Lets leave .2 of the data for testing, this is standard practice
    X_train, X_test, y_train, y_test = train_test_split(X.values, Y, test_size=0.2, random_state=42)

    # Z-normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Free to assign custom k values in KNN()
    knn = KNN()
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy:", accuracy)

    return predictions, y_test


predictions, y_test = LoadTrain('features_3_sec.csv')

def confusion_m(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.show()

confusion_m(y_test, predictions)
