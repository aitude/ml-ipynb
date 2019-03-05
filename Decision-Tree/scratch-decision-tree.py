"""Decision Tree Implementation From Sctrach in Python

This class implement Decision Tree algorithm using Information Gain method in python without using ml libraries.
"""

# Load Libraries

import numpy as np
import pandas as pd

class DecisionTree:

	# Formula - -pi * log(pi)
	# pi = number1/(number1 + number2)
	def calculate_entropy(self,num,denominator):
	     
	     pi = num/denominator
	     if pi == 0:
	     	# To avoid divided by zero when calculating np.log2
	     	return 0
	     else:	
	     	return -pi*np.log2(pi)

	# Calculate E(Target) - Entropy of Target
	def calculate_target_entropy(self,vector):
		entropy = 0
		values = vector.value_counts()
		total = len(vector)
		for value in values:
			entropy += self.calculate_entropy(value,total)
		return entropy	

	# Calculate E(Target | Attribute) - Entropy of Features
	def calculate_attribute_entropy(self,dataset,attribute,target):
		targets = dataset[target].unique()
		attribute_vector = dataset[attribute]
		
		total_samples = len(attribute_vector)
		properties = attribute_vector.unique()
		entropy = 0
		for prop in properties:
			prop_entropy = 0
			denominator = len(dataset[attribute][ dataset[attribute] == prop ])
			for target_class in targets:
				number = len(dataset[attribute][ dataset[attribute] == prop ][dataset[target] == target_class ])
				prop_entropy += self.calculate_entropy(number,denominator)

			p_attribute = denominator/total_samples
			entropy += p_attribute*prop_entropy
		return entropy

	# Calculate Information Gain of Features	
	# Formula = E(Target) - E(Target| Attribute)
	def calculate_information_gain(self,dataset,attribute,target):
		target_entropy = self.calculate_target_entropy(dataset[target])
		attribute_entropy = self.calculate_attribute_entropy(dataset,attribute,target)
		return target_entropy - attribute_entropy


	# Find out decision node by calulating max Information Gain
	def winner_attribute(self,df):
		
		information_gain = []
		target = df.keys()[-1]
		features =  df.keys()[:-1] # Exclude the last one.
		
		for feature in features: 
			information_gain.append(self.calculate_information_gain(df,feature,target))
		
		maximum_ig_index = np.argmax(information_gain)
		winner_feature = features[maximum_ig_index]
		return winner_feature

	# Split the dataset on decision node	
	def split_dataset(self,df,node,value):
		return df[df[node] == value].reset_index(drop=True)

	# Build Decision Tree
	def build_tree(self,df,tree=None):
		target_class = df.keys()[-1]
		node = self.winner_attribute(df)
		node_values= df[node].unique()
		if tree is None:
			tree= {}
			tree[node] = {}
		for value in node_values:
			subtable = self.split_dataset(df,node,value)
			subset_target_class = subtable[target_class].unique()
			if len(subset_target_class) == 1:
				tree[node][value] = subset_target_class[0]
			else:
				tree[node][value] = self.build_tree(subtable)

		return tree

	# Start training process. Ultimate goal is to make a decision tree.
	def fit(self,df):
		self.tree = self.build_tree(df)	


	# Traverse through decision tree.
	def traverse_tree(self,guess,tree):

	   prediction = ''
	   for node in tree.keys():
	   	value = guess[node]
	   	tree = tree[node][value]
	   
	   	if type(tree) is dict:
	   		prediction = self.traverse_tree(guess,tree)
	   	else:
	   		prediction = tree
	   		break

	   return prediction


	# Predict the class using Input values
	def predict(self,guess):

	   prediction = ''
	   tree = self.tree
	   prediction = self.traverse_tree(guess,tree)
	   return prediction			

# How to Use

# Sample Data
dataset = {'Taste':['Salty','Spicy','Spicy','Spicy','Spicy','Sweet','Salty','Sweet','Spicy','Salty'],
       'Temperature':['Hot','Hot','Hot','Cold','Hot','Cold','Cold','Hot','Cold','Hot'],
       'Texture':['Soft','Soft','Hard','Hard','Hard','Soft','Soft','Soft','Soft','Hard'],
       'Eat':['No','No','Yes','No','Yes','Yes','No','Yes','Yes','Yes']}

# Convert into Dataframe
dataframe = pd.DataFrame(dataset,columns=['Taste','Temperature','Texture','Eat'])

# Training the model
model = DecisionTree()

# Train the model
model.fit(dataframe)


"""
Problem - Should I eat if taste is salty, temperature is hot and texture is hard?
"""
data = {'Taste':'Salty','Temperature':'Hot','Texture':'Hard'}
guess = pd.Series(data)

# Predict class 
prediction = model.predict(guess)
#result
print(prediction)