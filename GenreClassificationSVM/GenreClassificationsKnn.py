import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=5):
        self.k = k
    
    # This approach differs from the SVM approach as we input x and y rather than take whole dataset as matrix
    def fit(self, x , y):
        self.X_train = x
        self.Y_train = y

    # Queue up distance calculations for each point in the training set
    def predictHelper(self, x):
        distances = [self.predictHelper(x) for x in x]

        # Gives original indices after sorting, so we get indices of closest three neighbors which are first k elements in sorted list
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        common = Counter(k_nearest_labels).most_common()
        return common

    # Computer calculates euclidean distance
    def predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
