"""Multiple Linear Regression Algorithm From Scratch Using Python

This class is implementing Multiple Linear Regression Algorithm without using machine learning libraries.

"""
#load libaries
import numpy as np

class MultipleLinearRegression:
    
    def __init__(self, learning_rate = .0001, loss_threshold = .0001):
        
        # Learning rate controls how much we are adjusting the coefficient with respect the loss error in each iteration.
        self.learning_rate = learning_rate
        
        # Minimum Loss Error for better prediction.
        self.loss_threshold = loss_threshold

    """ Prepare Dataset before start training
    
    This method append a scalar column matrix [1,1,1,....] into features dataset
    """

    def prepare_data(self,features):
        features = features.T
        # Add feature0 vector to the features matrix 
        feature0 = np.ones(len(features[0])).reshape(1,-1)
        features = np.concatenate((feature0, features)).T
        return features

    """Calculate Loss using Mean Squared Error Method
    
    This method calculates mean sqaured error (MSE).
    """
    def cost_function(self,X,Y,B):
        count = len(Y)
        loss = X.dot(B) - Y
        square = (loss) ** 2
        mean_square = sum(square) / (2 * count)
        return mean_square
    
    """Calculate New Coefficient using Gradient Descent Method 
    
    This method calculates new coefficient to minimize the loss error. 
    """
    def gradient_descent(self,X,Y,B, learning_rate):
        count = len(Y)
        Y_predict = X.dot(B)
        loss = Y_predict - Y
        gradient = (X.T).dot(loss) / count
        new_B = B - learning_rate * gradient
        return new_B
    
    """Calculate Appropriate Coefficient Values for Linear Regression equation
    
    This method calculates appropriate coefficient values by minimizing Mean Squared Error (MSE) 
    """
    def fit(self,features,target):
        
        features = self.prepare_data(features)
        coefficient = np.zeros(len(features[0]))
        learning_rate = self.learning_rate
        loss_error = self.cost_function(features,target,coefficient)
        iteration = 1;
        while loss_error > self.loss_threshold:
            coefficient = self.gradient_descent(features,target,coefficient, learning_rate)
            loss_error = self.cost_function(features,target,coefficient)
            iteration = iteration + 1
        
        self.coefficient = coefficient 
            
    """Predict Target Values Using Features
    
    This method predict the target value using Linear Regression equation:
    
    target = bo + b1 * feature1 + b2 * feature2 + .....

    Assume feature0 = 1 so
    
    target = bo * feature0 + b1 * feature1 + b2 * feature2 + .....

    In Matrix Form:

    target (Y) = Coefficient(B) * Features (X)

    Here - 

    Coefficient is calculated using fit() method
    Feature0 is appended to Features List using prepare_data() method

    """
    def predict(self,features):
        features = self.prepare_data(np.array(features))
        predicted = features.dot(self.coefficient)
        return predicted

# Multiple Features Variables

feature1 = [1,2,3,4,5,6,7,8,9,10]
feature2 = [10,20,30,40,50,60,70,80,90,100]

# Target Variable
target = [101,202,303,404,505,606,707,808,909,1010]

# So, target = ( 0 * feature0 + 1 * feature1 + 10 * features2 ) so coefficient are bo = 0, b1 = 1 and b2 = 10 
# but we have to let do these calculation to this multiple linear regression class.

# A numpy array containing all features
features = np.array([feature1,feature2]).T

# Initialize MultipleLinearRegression model
model = MultipleLinearRegression(learning_rate = .0001,  loss_threshold = .0001)

# Train the model
model.fit(features,target)    

# what is value of y if feature1 = 11 and feature2 = 110. You can guess that it should be 1111.
y_predicted = model.predict([[11,110]])# 
print("Predicted Value %s" % (y_predicted))        