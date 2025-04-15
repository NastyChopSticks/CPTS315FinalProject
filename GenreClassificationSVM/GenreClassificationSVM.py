from pyexpat import features
from xml.etree.ElementPath import xpath_tokenizer
import pandas as pd
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def get_data_matrix(file):
    #used for printing features matrix
    data_matrix = pd.read_csv(file)
    return data_matrix
   

def get_all_features(file):
    #we need to create a features matrix that extract all usable features from the matrix. Looking at the features matrix we will need to exlude a couple of "features" that arent actually features.
    #mainly the filename, length, and label. which is located at column index 0,1 and 59

    data_matrix = get_data_matrix(file)
    features_columns = [col for col in data_matrix.columns if col not in ["filename", "length", "label"]]
    features_matrix = data_matrix[features_columns]
    return features_matrix
    
            
def get_all_labels(file):
    #get all labels and store into a matrix
     data_matrix = get_data_matrix(file)
     labels_matrix = data_matrix["label"]
     return labels_matrix


def get_top36_features_30_sec():
    X_training = get_all_features("features_30_sec.csv")
    Y_training = get_all_labels("features_30_sec.csv")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_training)
    model = SVC(kernel='linear')
    rfe = RFE(estimator=model, n_features_to_select=36)  
    rfe.fit(X_scaled, Y_training)  # X_scaled = your scaled feature matrix
    top36_indices = rfe.get_support(indices=True)
    X_top36 = X_scaled[:, top36_indices]
    return X_top36
    
def get_top52_features_3_sec():
    X_training = get_all_features("features_3_sec.csv")
    Y_training = get_all_labels("features_3_sec.csv")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_training)
    model = SVC(kernel='linear')
    rfe = RFE(estimator=model, n_features_to_select=52, step=5)  # step=5 speeds it up
    rfe.fit(X_scaled, Y_training)  # X_scaled = your scaled feature matrix
    top52_indices = rfe.get_support(indices=True)
    X_top52 = X_scaled[:, top52_indices]
    return X_top52

def find_top_features_3_sec():
    scores = []
    k = range(2,58, 2)
    X_training = get_all_features("features_3_sec.csv")
    Y_training = get_all_labels("features_3_sec.csv")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_training)
    for k in k:
        model = SVC(kernel='linear')
        rfe = RFE(estimator=model, n_features_to_select=k, step=5)
        X_k = rfe.fit_transform(X_scaled, Y_training)
        score = cross_val_score(model, X_k, Y_training, cv=5).mean()
        scores.append(score)
        print(f"Top {k} features -> Accuracy: {score:.4f}")

def find_top_features_30_sec():
    scores = []
    k = range(2,58, 2)
    X_training = get_all_features("features_30_sec.csv")
    Y_training = get_all_labels("features_30_sec.csv")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_training)
    for k in k:
        model = SVC(kernel='linear')
        rfe = RFE(estimator=model, n_features_to_select=k)
        X_k = rfe.fit_transform(X_scaled, Y_training)
        score = cross_val_score(model, X_k, Y_training, cv=5).mean()
        scores.append(score)
        print(f"Top {k} features -> Accuracy: {score:.4f}")


def test_without_top_features_30_sec():
    Y_training = get_all_labels("features_30_sec.csv")
    X_training = get_all_features("features_30_sec.csv")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_training)
    model = SVC(kernel='linear')
    scores = cross_val_score(model, X_scaled, Y_training)
    print("Test without top features for 30 second clips")
    print("Accuracy per fold:", scores)
    print("Mean accuracy:     ", np.mean(scores))
    print("Standard deviation:", np.std(scores))

def test_without_top_features_3_sec():
    Y_training = get_all_labels("features_3_sec.csv")
    X_training = get_all_features("features_3_sec.csv")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_training)
    model = SVC(kernel='linear')
    scores = cross_val_score(model, X_scaled, Y_training)
    print("Test without top features for 3 second clips")
    print("Accuracy per fold:", scores)
    print("Mean accuracy:     ", np.mean(scores))
    print("Standard deviation:", np.std(scores))

def test_with_top_features_30_sec():
    Y_training = get_all_labels("features_30_sec.csv")
    model = SVC(kernel='linear')
    scores = cross_val_score(model, get_top36_features_30_sec(), Y_training)
    print("Test with top features for 30 second clips")
    print("Accuracy per fold:", scores)
    print("Mean accuracy:     ", np.mean(scores))
    print("Standard deviation:", np.std(scores))


def test_with_top_features_3_sec():
    Y_training = get_all_labels("features_3_sec.csv")
    model = SVC(kernel='linear')
    scores = cross_val_score(model, get_top52_features_3_sec(), Y_training)
    print("Test with top features for 3 second clips")
    print("Accuracy per fold:", scores)
    print("Mean accuracy:     ", np.mean(scores))
    print("Standard deviation:", np.std(scores))



def main():
    test_without_top_features_3_sec()
    test_with_top_features_3_sec()
    test_without_top_features_30_sec()
    test_with_top_features_30_sec()
    


if __name__ == "__main__":
    main()