""" K-Nearest Neighbors (KNN) Algorithm Implementation from sctrach in Python

This class implement knn algorithm without using any machine learning library

"""

# Load Libraries
import numpy as np
import math


class KNearestNeighbors:

	# Inititalize Number of Nearest Neighbor
	def __init__(self, k=1):
		self.k = k

	# Find euclidean distance between two vectors
	def euclidean_distance(self,feature,target):
		length = len(target)
		distance = 0
		for i in range(0,length):
			distance += pow( (feature[i] - target[i]),2)
		return math.sqrt(distance)

	"""This algorithm is consider as Lazy algorithm because no training step needed
	in this algorithm so this fit method just setup featuers and classes to process in predict method.
	
	"""

	def fit(self,features,classes):
		self.features = features
		self.classes = classes


	"""
	Find class of the target value
	"""
	def predict(self,target):

		from collections import Counter

		eu_distances = []

		k = self.k
		features = self.features
		classes = self.classes

		for i,element in enumerate(features):

			# Calculate Euclidean Distance
			eu_distance = self.euclidean_distance(element,target)
			
			eu_distances.append( (element,eu_distance,classes[i]) )

		""" 
		eu_distances is a list of touple consist of Element, Euclidean Distance from target and it's class
		"""

		# first sort eu_distances by calculted Euclidean Distance
		eu_distances.sort(key=lambda x:x[1])

		# then get first k elements
		k_sortest = eu_distances[:k]

		classes = []

		for element in k_sortest:
			classes.append(element[2])

		# Find number of occurance for each class	
		occurance = Counter(classes)

		# Pick most occured class or we can say that this class is mosted voted by nearest neighbor
		most_voted = occurance.most_common(1)
		
		return most_voted[0][0]
		


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

target = [3.5,3]

# features
features = number_classes[:,0:2].astype(float)

# classes
classes = number_classes[:,2]

# k is number of nearest neighbor to be considered to predict the class.
model = KNearestNeighbors(k=3)

# Train the model
model.fit(features,classes)

# Predict class 
prediction = model.predict(target)

#result
print("Number %s is %s" %(target,prediction))