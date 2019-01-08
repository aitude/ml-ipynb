"""Linear Regression Algorithm From Scratch Using Python

This class is implementing Linear Regression Algorithm without using machine learning libraries.

"""
from math import pow

class SimpleLinearRegression:

    """ Mean Calculation
    
    Mean is sum of all elements divided by total elements in the series
    """
    def calculate_mean(self, series):
        series_sum = sum(series)
        series_length = len(series)
        mean = series_sum / float(series_length)
        return mean

    """Variance Calculation
    
    Variance is sum of square of distance of each elements from its mean, divided by total elements.
    """
    def calculate_variance(self, series):
        mean = self.calculate_mean(series)
        series_length = len(series)
        squared_difference = [pow(value - mean, 2) for value in series]
        variance = sum(squared_difference) / float(series_length-1)
        return variance

    """Covariance Calculation
    
    Covariance is mean value of the multiply of the deviaions of two variance.
    """
    def calculate_covariance(self, series1, series2):
        series1_mean = self.calculate_mean(series1)
        series2_mean = self.calculate_mean(series2)
        series1_length = len(series1)
        covariance = 0.0
        for i in range(0, series1_length):
            covariance = covariance + \
                (series1[i] - series1_mean) * (series2[i] - series2_mean)
        return covariance / float(series1_length - 1)

    """Slope of linear line
    
    Formula is covariance divided by invariance.
    """
    def b1_coefficient(self, series1, series2):
        covariance = self.calculate_covariance(series1, series2)
        variance = self.calculate_variance(series1)
        b1 = covariance / float(variance)
        return b1

    """Intersect of Linear line
    
    Formula is bo = y-b1*x
    """
    def b0_coefficient(self, series1, series2):
        series1_mean = self.calculate_mean(series1)
        series2_mean = self.calculate_mean(series2)
        b1 = self.b1_coefficient(series1, series2)
        b0 = series2_mean - b1 * series1_mean
        return b0

    """Train the model 
    
    Actually we're calculating b1 and bo coefficient values here.
    """
    def fit(self, features, targets):
        self.b0 = self.b0_coefficient(features, targets)
        self.b1 = self.b1_coefficient(features, targets)

    """ Predict the value
    
    Formula is y = b0 + b1*x

    We have calculated bo and b1 already in the fit method.
    """
    def predict(self, features):
        b0, b1 = self.b0, self.b1
        predicted_values = b0 + b1 * features
        return predicted_values

# Features Variable
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Target Variable
Y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Initialize SimpleLinearRegression model
model = SimpleLinearRegression()

# Train the model
model.fit(X, Y)

# what is value of y if x = 40. You can guess that it should be 400.
print(model.predict(40))
