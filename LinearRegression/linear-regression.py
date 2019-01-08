# load libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# load dataset
dataset = pd.read_csv('../dataset/house_prices.csv')

# Single Feature Variable - Size of the Houses
size_data = dataset['size']

# Target/Dependent Variable - Price of the Houses
price_data = dataset['price']

# Size and Target are dataframes. We have to convert them into Array to be used as Training dataset. 
# and then use eshape function to convert array.shape(1,n) to array.shape(n,1) so each independent variable has own row.

size = np.array(size_data).reshape(-1,1)
price = np.array(price_data).reshape(-1,1)

# Train the Model
model = LinearRegression()
model.fit(size,price)

# Predict Price
price_predicted = model.predict(size)

# Plot the result
plt.scatter(size,price, color="green")
plt.plot(size,price_predicted, color="red")
plt.title("Linear Regression")
plt.xlabel("House Size")
plt.ylabel("House Price")