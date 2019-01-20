# Import Libraries
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Sample Data
number_classes = np.array([

	[1,3,'red'],
	[2,4,'red'],
	[2,1,'red'],
	[3,1,'red'],
	[3,3,'red'],
	[4,2,'green'],
	[5,3,'green'],
	[4,4,'green'],
	[5,1,'green'],
	
	])

"""

Problem - What is color if number are 3.5 and 3?

"""

target = [[3.5,3]]

# Features
features = number_classes[:,0:2].astype(float)

# Classes
classes = number_classes[:,2]

# Create and fit a nearest-neighbor classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the Model
knn.fit(features, classes) 

# Predict class
prediction = knn.predict(target)

#output
print("Number %s is %s" %(target,prediction))