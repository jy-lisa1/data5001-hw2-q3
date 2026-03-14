import numpy as np

class DecisionTree:
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            """Initialize a node in the decision tree"""
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value   

    def __init__(self, max_depth=5, criterion='gini', min_samples_leaf=2, min_samples_split=2):
        """Initialize Decision Tree model with specified hyperparameters"""
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf 
        self.root = None

    def gini(self, y):
        """Compute gini impurity for set of labels"""
        classes = np.unique(y)
        impurity = 1
        for c in classes:
            p = np.sum(y == c) / len(y)
            impurity -= p ** 2
        return impurity
    
    def entropy(self, y):
        """Compute entropy of set of labels"""
        classes = np.unique(y)
        entropy = 0
        for c in classes:
            p = np.sum(y == c) / len(y)
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    def impurity(self, y):
        """Return cost function based on which one was chosen"""
        if self.criterion == "entropy":
            return self.entropy(y)
        return self.gini(y) #gini is the default
    
    def best_split(self, X, y):
        """Find best feature and threshold split data"""
        best_feature = None
        best_threshold = None
        best_impurity = float("inf")

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

                imp_left = self.impurity(y_left)
                imp_right = self.impurity(y_right)

                weighted_impurity = (
                    len(y_left)/len(y) * imp_left +
                    len(y_right)/len(y) * imp_right
                )

                if weighted_impurity < best_impurity:
                    best_impurity = weighted_impurity 
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold
    
    def most_common_label(self, y):
        """Determine most common class label in set of labels"""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
    
    def build_tree(self, X, y, depth):
        """Recursively build decision tree"""
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            return self.Node(value=self.most_common_label(y))

        feature, threshold = self.best_split(X, y)

        if feature is None:
            return self.Node(value=self.most_common_label(y))

        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold

        if np.sum(left_idx) < self.min_samples_leaf or np.sum(right_idx) < self.min_samples_leaf:
            return self.Node(value=self.most_common_label(y))

        left = self.build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self.build_tree(X[right_idx], y[right_idx], depth + 1)

        return self.Node(feature, threshold, left, right)
    
    def fit(self, X, y):
        """Train decision tree model using provided dataset"""
        # convert pandas to numpy if needed
        self.X_array = np.array(X)
        self.y_array = np.array(y)
        self.root = self.build_tree(self.X_array, self.y_array, 0)

    def predict_sample(self, x, node):
        """Predict class label for single sample by traversing tree"""
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

    def predict(self, X):
        """Predict class labels for multiple input samples"""
        X = np.array(X)
        predictions = []
        for x in X:
            predictions.append(self.predict_sample(x, self.root))
        return np.array(predictions)

    
