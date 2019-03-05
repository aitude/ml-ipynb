"""Decision Tree Classification Implementation in Python

This class implement Decision Tree Classification algorithm in python using scikit library
"""

# Load Libraries

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import accuracy_score

# How To Use

# Sample Data - (Taste, Temperature, Texture,Eat)
dataset = np.array([
		   ['Salty','Hot','Soft','No'],
		   ['Spicy','Hot','Soft','No'],
		   ['Spicy','Hot','Hard','Yes'],
		   ['Spicy','Cold','Hard','No'],
		   ['Spicy','Hot','Hard','Yes'],
		   ['Sweet','Cold','Soft','Yes'],
		   ['Salty','Cold','Soft','No'],
		   ['Sweet','Hot','Soft','Yes'],
		   ['Spicy','Cold','Soft','Yes'],
		   ['Salty','Hot','Hard','Yes'],
		   ])

# We need to convert categorical data into numerical data for process.
enc = preprocessing.OrdinalEncoder()
enc.fit(dataset)

data = enc.transform(dataset)

features = data[:,0:-1]
targets = data[:,-1]

class_mapping = {}

for num, label in zip((data[:,-1].astype('int')), (dataset[:,-1])):
    class_mapping[num] = label

feature_train, feature_test, target_train, target_test = train_test_split(features, targets,test_size=.10)

# Setup Decision Tree Model
model = DecisionTreeClassifier(criterion='entropy')

# Train the Model
model.fitted  = model.fit(feature_train, target_train)

"""
Problem - Should I eat if taste is salty, temperature is hot and texture is hard?
"""
guess = np.array([['Salty','Hot','Hard']])

predictions = model.fitted.predict( enc.transform(guess) )
predictions_labels = [class_mapping[key] for key in predictions]
print(predictions_labels)

