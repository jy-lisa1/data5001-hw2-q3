import numpy as np 

class KNN:
    def __init__(self):
       pass

    
    def calculate_distance(self,x1, x2):
        if self.distance == "euclidean":
            return np.linalg.norm(x1 - x2)
        elif self.distance == "manhattan":
            return np.sum(np.abs(x1 - x2))
        elif self.distance == "minkowski":
            return np.power(np.sum(np.power(np.abs(x1 - x2), 3)), 1/3)
    
    def get_neighbours(self, x):
        distances = [] #store the distances between x and other points

        #loop through all the training examples
        for x_t in self.x_train:
            distance = self.calculate_distance(x, x_t)
            distances.append(distance)
        
        #sort distances in ascending order and return the closest k
        neighbours_indexs = np.argsort(distances)[: self.k_neighbours]
        
        return neighbours_indexs
        

    def fit(self, x_train, y_train, k, distance):
        self.k_neighbours = k
        self.distance = distance
        # convert to numpy arrays for easier calculations
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

    def predict(self, x_test):
        
        y_preds = [] #store the predictions

        x_test = np.array(x_test)
        # Loop through all the test examples
        for x in x_test:
            # Get neighbours
            neighbours_indexs = self.get_neighbours(x)

             #find the most common label amoung the neighbours
            labels = self.y_train[neighbours_indexs]                  
            labels = list(labels)
            prediction = max(labels, key=labels.count)
        
            # Append the prediction to the predictions list
            y_preds.append(prediction)

        return np.array(y_preds)

