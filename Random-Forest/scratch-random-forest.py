"""Random Forest Classifier Implementation From Sctrach in Python

This class implement Random Forest Classifier algorithm method in python without using ml libraries.
"""

# Load Libraries

import numpy as np
import pandas as pd

'''A class to build Decision Tree 

This class is used to create the decision tree.
'''
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

'''A class to Implement Random Forest Clasifier from scratch

This class divides a dataset into small dataset and create a decision tree on each subset.
'''
class RandomForest():

	def __init__(self,n_trees,n_features,sample_sz,depth=10,min_leaf=5):
		np.random.seed(12)

		self.n_features = n_features
		self.n_trees = n_trees
		self.sample_sz, self.depth, self.min_leaf = sample_sz,depth,min_leaf
		

	def fit(self, feature, target):
		
		# Decide number of fetures in each subset
		if self.n_features == 'sqrt':
			self.n_features = int( np.sqrt(features.shape[1]))
		elif self.n_features == 'log2':
			self.n_features = int(np.log2(features.shape[1]))
	
		self.x = features
		self.y = target
		self.trees = [self.create_tree() for i in range(self.n_trees)]	

	def create_tree(self):
		# Divide databse into small dataset
		idxs = np.random.permutation(len(self.y))[:self.sample_sz]
		f_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
		y = self.y[idxs].reshape(-1,1)
		x = self.x[idxs]
		dataset = np.concatenate((x,y),axis=1)
		
		# Convert dataset into dataframe
		dataframe = pd.DataFrame(dataset)
		
		# Use DecisionTree class to create a decision for this small dataset
		model = DecisionTree()
		model.fit(dataframe)
		return model

	def predict(self,x):
		prediction = []
		# Predict possible class using number of decision tree created on small subset of dataset.
		for dt in self.trees:
			prediction.append(dt.predict(x))

		return prediction
				
# How to Use

# Sample dataset - Taste, Temperature, Texture and eat 
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

features= dataset[:,0:-1]

target= dataset[:,-1]

model = RandomForest(5,'log2',6);

# Training the model
model.fit(features,target)

"""
Problem - Should I eat if taste is salty, temperature is hot and texture is hard?
"""
data = {'Taste':'Spicy','Temperature':'Cold','Texture':'Soft'}
guess = pd.Series(data)

# Predict class 
prediction = model.predict(guess)

#result - Maximum occurance of class in the prediction list is the winner.
most_voted = max(set(prediction), key = prediction.count)

print("Should I eat if taste is %s, temperature is %s and texture is %s ? - %s" % (data['Taste'],data['Temperature'],data['Texture'],most_voted))
