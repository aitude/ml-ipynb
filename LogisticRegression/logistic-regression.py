"""Logistic Regression Implementation Using Python

This example demonstrate logistic regression implementation.

"""

import numpy as np
from sklearn.linear_model import LogisticRegression

# Features
features = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

# Target - if class 0 if number <=5 and class 1 if number >5
target = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

model = LogisticRegression()

# Train the Model
model.fit(features,target)

guess = np.array([[3], [12], [13], [1], [2], [8], [5], [18], [4], [20]])

# find out class for guess values
print(model.predict(guess, .60))