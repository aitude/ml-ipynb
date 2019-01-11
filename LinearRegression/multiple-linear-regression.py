#Import Libraries
from sklearn.linear_model import LinearRegression
import numpy as np

# Multiple Features Variables
feature1 = [1,2,3,4,5,6,7,8,9,10]
feature2 = [10,20,30,40,50,60,70,80,90,100]

# Target Variable
target = [101,202,303,404,505,606,707,808,909,1010]

# A numpy array containing all features
features = np.array([feature1,feature2]).T

# Initialize LinearRegression model
model = LinearRegression()

# Train the model
model = model.fit(features, target)

# what is value of y if feature1 = 11 and feature2 = 110. You can guess that it should be 1111.
y_predicted = model.predict([[11,110]])

print("Predicted Value %s" % (y_predicted))        