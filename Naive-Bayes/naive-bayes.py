"""Naive Bayes Classification Implementation in Python

This class implement Naive Bayes Classification algorithm in python using scikit library
"""

# Load Libraries

import numpy as np
from sklearn.naive_bayes import GaussianNB

# How To Use

# Sample Data - (weight (kg), Height (cm), gender)
sample_data = np.array([
	[70,175,'male'],
	[60,140,'female'],
	[80,185,'male'],
	[75,180,'male'],
	[65,150,'female'],
	[70,155,'female'],
	[75,160,'female'],
	[85,195,'male'],
	[55,170,'female'],
	[65,175,'female'],
	])

# Setup Gaussian Naive Bayes Model

model = GaussianNB()


# Features
features = sample_data[:,0:-1].astype('float')

# Target
target = sample_data[:,-1]

# Train the Model
fittedModel = model.fit(features, target)

# find out gender if weight is 50kg and height is 172 cm.
predictions = fittedModel.predict([[50,172]])

print(predictions)