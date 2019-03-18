"""Random Forest Classifier Implementation in Python

This class implement Decision Tree Classification algorithm in python using scikit library
"""

# Load Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets
import numpy as np
from sklearn import preprocessing


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

feature_train, feature_test, target_train, target_test = train_test_split(features, targets)

# Setup Random Forest Model
model = RandomForestClassifier(n_estimators = 1000, max_features = 'sqrt')

# Train the Model 
fitted_model = model.fit(feature_train, target_train)

"""
Problem - Should I eat if taste is salty, temperature is hot and texture is hard?
"""
guess = np.array([['Salty','Hot','Hard']])

# Result
predictions = fitted_model.predict( enc.transform(guess) )
predictions_labels = [class_mapping[key] for key in predictions]
print("Should I eat if taste is %s, temperature is %s and texture is %s ? - %s" % (guess[0][0],guess[0][1],guess[0][2],predictions_labels[0]))

