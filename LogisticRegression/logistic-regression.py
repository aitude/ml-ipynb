"""Logistic Regression Implementation From Sctrach in Python

This class implement Logistic Regression algorithm in python without using ml libraries.
"""

# Load Libraries
import numpy as np


class LogisticRegression:

    """Initialize Model Properties

    Set learning rate and number of iterations for training the model.
    """

    def __init__(self, learning_rate=.001, iteration=10000):
        self.learning_rate = learning_rate
        self.iteration = iteration

    """ Predict the target using Sigmoid Function
    
    This method is used to calculate the target value using features and coefficients.
    
    """

    def hypothesis(self, features, coefficient):
        y = np.dot(features, coefficient)
        return 1.0/(1+np.exp(-y))

    """Calculate Loss Error
    
    Formula:
    
    loss_error = -Y * log( h(x) ) - (1-Y) * log( 1- h(x) )

    Here -

    h(x) - Value return by Sigmoid function.

    """
    def cost_function(self, features, target, coefficient):
        hypothesis = self.hypothesis(features, coefficient)
        part1 = target * np.log(hypothesis)
        part2 = (1-target).T * np.log((1-hypothesis))
        loss_error = np.mean(-part1 - part2)
        return loss_error

    """Calculate gradient decent to minimize the output given by cost function
    
    Formula:

    gradient = features * (h(x) - target) 
    """

    def gradient_decent(self, features, target, coefficient):
        size = target.size
        hypothesis = self.hypothesis(features, coefficient)
        gradient = np.dot(features.T, (hypothesis - target))
        return gradient/size

    """Append feature0 matrix to the features
    
    This function append feature0 = 1 matrix to the features.
    """
    def add_intercept(self, features):
        feature0 = np.ones((features.shape[0], 1))
        features = np.concatenate((feature0, features), axis=1)
        return features

    """Calculate Appropriate Coefficient Values
    
    This method calculates appropriate coefficient values by minimizing the cost function
    """

    def fit(self, features, target):
        features = self.add_intercept(features)
        coefficient = np.zeros(features.shape[1])
        target = target
        loss_error = 10000
        iteration = 1
        while iteration < self.iteration:
            gradient = self.gradient_decent(features, target, coefficient)
            coefficient = coefficient - self.learning_rate * gradient
            loss_error = self.cost_function(features, target, coefficient)
            iteration = iteration + 1
        self.coefficient = coefficient

    """ Calculate Probablity using Sigmoid Function
    
    This method is used to calculate the probablity using Sigmoid function.
    """
    def predict_prob(self, features):
        features = np.array(features)
        features = self.add_intercept(features)
        # y = 1/1+e(-BX)
        y = self.hypothesis(features, self.coefficient)
        return y

    """Predict Target Values Using Features
    
    We assume that class is 1 (True) if probablity given by predict_prob is greater than thresold 
    """
    def predict(self, features, threshold):
        return self.predict_prob(features) >= threshold


# Features
features = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

# Target - if class 0 if number <=5 and class 1 if number >5
target = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Setup LogisticRegression Model
model = LogisticRegression(learning_rate=.1, iteration=10000)

# Train the Model
model.fit(features, target)

guess = np.array([[3], [12], [13], [1], [2], [8], [5], [18], [4], [20]])

# find out class for guess values
print(model.predict(guess, .60))
