"""Naive Bayes Classification Implementation From Sctrach in Python

This class implement Naive Bayes Classification algorithm in python without using ml libraries.
"""

# Load Libraries

import numpy as np
import math

class NaiveBayes:

	"""Calculate Probablity if values are continous using Gaussian Naive Bayes Formula
	
	This function calculate probablity for continous values
	"""

	def gaussian_formula(self,feature, mean, standard_deviation):
		exponent = math.exp(-(math.pow(feature-mean,2)/(2*math.pow(standard_deviation,2))))
		return (1/(math.sqrt(2*math.pi)*standard_deviation)) * exponent

	"""Calculate Mean
	
	This function calculate means of numbers
	"""
	def calculate_mean(self,numbers):
		return sum(numbers)/(float)(len(numbers))

	"""Calculate Standard Deviation
	
	This function calculate standard deviation of numbers
	"""
	def calculate_standard_deviation(self,numbers):
		mean = self.calculate_mean(numbers)

		difference = 0

		for num in numbers:
			difference += math.pow((num - mean),2)

		return math.sqrt(difference/(float)(len(numbers) -1 ))

	"""Calculate Mean and Standard Deviation for each feature

	This function create mean and standard deviation for each feature with respect to classes
	"""
	def fit(self,data):

		probabilities = []

		# Group Rows By Class
		# Goal - array( 'male' => numpy array, 'female' => numpy array)
		separation = {}
		for row in data:

			if row[-1] not in separation:
				separation[row[-1]]  = []
			separation[row[-1]].append(row[0:-1].astype('float'))

		
		""" Calculate Mean and Standard Deviation for each feature 
		
		Goal - array( 'male' => [

			[weight mean, weight standard deviation]
			[height mean, height standard deviation]

			],

			'female' => [

			[weight mean, weight standard deviation]
			[height mean, height standard deviation]

			],
		)
		"""
		class_mean_stdev = {}	
		for class_name,rows in separation.items():
			
			if class_name not in class_mean_stdev:
				class_mean_stdev[class_name] =[]

			for features in zip(*rows):
				class_mean_stdev[class_name].append([self.calculate_mean(features),self.calculate_standard_deviation(features)])

		self.class_mean_stdev = class_mean_stdev

	""" Predict Class of Input Values
	
	This function predict class of input values by comparing probablity of each class. Class with highest Probablity is the winner.
	"""
	def predict(self,input):
		predictions = []
		class_means_stdev = self.class_mean_stdev
		
		# Find out Probablity for each class
		# and Predicted class will have highest probablity.
		all_probabilities = []
		for row in input:
			probabilities = []
			for class_name, features in class_means_stdev.items():
				probability = 1
				for i in range(len(row)):
					mean, standard_deviation = features[i]
					probability *= self.gaussian_formula(row[i],mean,standard_deviation)
				probabilities.append([class_name,probability])
			all_probabilities.append(probabilities)	

		predictions = []
		for probability in all_probabilities:
			probability.sort( key=lambda x: x[1], reverse = True )
			predicted_class_name, probability_score = probability[0]
			predictions.append(predicted_class_name)
		
		return predictions	
	

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

# Setup Naive Bayes Model
model = NaiveBayes()

# Train the Model
model.fit(sample_data)

# find out gender if weight is 50kg and height is 172 cm.
print(model.predict([[50,172]]))