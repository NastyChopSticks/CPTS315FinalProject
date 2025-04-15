from pyexpat import features
from xml.etree.ElementPath import xpath_tokenizer
import pandas as pd
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler



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
    #this function returns the top 36 features from the features csv
    X_training = get_all_features("features_30_sec.csv")
    Y_training = get_all_labels("features_30_sec.csv")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_training)
    model = SVC(kernel='linear')
    rfe = RFE(estimator=model, n_features_to_select=36)  
    rfe.fit(X_scaled, Y_training)
    top36_indices = rfe.get_support(indices=True)
    X_top36 = X_scaled[:, top36_indices]
    return X_top36
    
def get_top52_features_3_sec():
    #returns top 52 features from the features csv
    X_training = get_all_features("features_3_sec.csv")
    Y_training = get_all_labels("features_3_sec.csv")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_training)
    model = SVC(kernel='linear')
    rfe = RFE(estimator=model, n_features_to_select=52, step=5) 
    rfe.fit(X_scaled, Y_training)  
    top52_indices = rfe.get_support(indices=True)
    X_top52 = X_scaled[:, top52_indices]
    return X_top52

def find_top_features_3_sec():
    #this function and the other one for 30 seconds uses a method call RFE, recursive feature elimation. It takes an input of n features and caclulates which features contribute the most to the accuracy
    #This function will loop through all features and output the accuracy. RFE ideally helps reduce noisy data, although the results show that RFE for the 3 second features doesnt boost accuracy a whole lot.
    #using this function I found the top 52 features for the 3 second features
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
    #same as the 3 second except the best accuracy split was the top 36 features
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
    #We also use a cross validation training method which splits the data into k equal folds. Then tests 1 fold and trains on 4 for every fold.
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
    #We also use a cross validation training method which splits the data into k equal folds. Then tests 1 fold and trains on 4 for every fold.
    scores = cross_val_score(model, X_scaled, Y_training)
    print("Test without top features for 3 second clips")
    print("Accuracy per fold:", scores)
    print("Mean accuracy:     ", np.mean(scores))
    print("Standard deviation:", np.std(scores))

def test_with_top_features_30_sec():
    Y_training = get_all_labels("features_30_sec.csv")
    model = SVC(kernel='linear')
    #We also use a cross validation training method which splits the data into k equal folds. Then tests 1 fold and trains on 4 for every fold.
    scores = cross_val_score(model, get_top36_features_30_sec(), Y_training)
    print("Test with top features for 30 second clips")
    print("Accuracy per fold:", scores)
    print("Mean accuracy:     ", np.mean(scores))
    print("Standard deviation:", np.std(scores))


def test_with_top_features_3_sec():
    Y_training = get_all_labels("features_3_sec.csv")
    model = SVC(kernel='linear')
    #We also use a cross validation training method which splits the data into k equal folds. Then tests 1 fold and trains on 4 for every fold.
    scores = cross_val_score(model, get_top52_features_3_sec(), Y_training)
    print("Test with top features for 3 second clips")
    print("Accuracy per fold:", scores)
    print("Mean accuracy:     ", np.mean(scores))
    print("Standard deviation:", np.std(scores))



def main():
    #also, each time we run a model on the features data set, it is required that we scale them. basically, the features that were provided are on a large scale. Some data it much smaller amd some is much larger.
    #this means our svm algorithm is going to be UBER slow. To fix this, or at least help fix this, we scale the data so it becomes more normalized. The algorithm not only becomes faster but also has increased accuracy

    #the results show hoesntly quite poor results for the SVM. With a highest accuracy of 70%. 
    #Upon doing some research it seems this data set is quite noisy. Some of the audio clips are intros and contain silence. And SVM is quite sensitive to this noisy data. Which may be one reason why the results arent so good
    #Also, most research papers I read used nueral networks and deep learning. And pretty much all of them extracted features from the data set themeselves. If we were to do the same we would likely see an increase in our results.
    test_without_top_features_3_sec()
    test_with_top_features_3_sec()
    test_without_top_features_30_sec()
    test_with_top_features_30_sec()
    


if __name__ == "__main__":
    main()