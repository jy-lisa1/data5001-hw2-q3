import numpy as np

class DecisionTree:
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value   

    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    def gini(self, y):
        classes = np.unique(y)
        impurity = 1
        for c in classes:
            p = np.sum(y == c) / len(y)
            impurity -= p ** 2
        return impurity
    
    def best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gini = float("inf")

        n_samples, n_features = X.shape

        for feature in range(n_features):

            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:

                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold

                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue

                y_left = y[left_idx]
                y_right = y[right_idx]

                gini_left = self.gini(y_left)
                gini_right = self.gini(y_right)

                weighted_gini = (
                    len(y_left) / len(y) * gini_left
                    + len(y_right) / len(y) * gini_right
                )

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold
    
    def most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
    
    def build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            leaf_value = self.most_common_label(y)
            return self.Node(value=leaf_value)

        feature, threshold = self.best_split(X, y)

        if feature is None:
            return self.Node(value=self.most_common_label(y))

        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold

        left = self.build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self.build_tree(X[right_idx], y[right_idx], depth + 1)

        return self.Node(feature, threshold, left, right)
    
    def fit(self, X, y):
        # convert pandas to numpy if needed
        self.X_array = np.array(X)
        self.y_array = np.array(y)
        self.root = self.build_tree(self.X_array, self.y_array, 0)

    def predict_sample(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            predictions.append(self.predict_sample(x, self.root))
        return np.array(predictions)


    
