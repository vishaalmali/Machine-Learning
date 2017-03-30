import numpy as np

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

# Perceptron Algorithm
class AdalineGD(object):


	# Initialize the perceptron method by defining the learning and number of iterations

	def __init__(self, eta = .01, n_iter = 10):

		self.eta = eta
		self.n_iter = n_iter

	def fit(self, X, y):
	
		# Fit training data

		# Parameters
		# -----------------
		# X : {array-like}, shape = [n_samples, n_features]
		# 	X is a matrix with the number of rows equal to the number of samples, and the number 
		# 	of columns equal to the number of features the data set has.

		# y : A column vector of the target values we are trying to hit
		

		# This defines the weights to be all zeros with a size of 1 plus number of features of the data set.
		self.w_ = np.zeros(1+X.shape[1])
		self.cost_ = []

		#This is the for loop that updates the weights after each epoch
		for i in range(self.n_iter):
			errors = 0
			for xi, target in zip(X, y):

				output = self.net_input(X)
				errors = (y - output)
				self.w_[1:] += self.eta * X.T.dot(errors)
				self.w_[0] += self.eta * errors.sum()
				cost = (errors**2).sum() / 2.0
				self.cost_.append(cost)
		return self


	def net_input(self, X):
		"""Calculate net input"""
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def activation(self, X):
		"""Computer linear activation"""
		return self.net_input(X)

	def predict(self, X):
		"""Return class label after unit step"""
		return np.where(self.activation(X) >= 0.0, 1, -1)

# Read in Iris data set
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# Initialize the target values
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# Initialize the X matrix (rows = number of samples | cols = number of features)
X = df.iloc[0:100, [0, 2]].values

# Plot visual decision boundary
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution = .02):

	# Setup marker generator and color map 
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	# Plot the decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contour(xx1, xx2, Z, alpha = .4, cmap = cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	# Plot class samples
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha = .8, c = cmap(idx), marker = markers[idx], label = cl)

# This standardizes the feature set 
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:,0].mean()) / X[:,0].std()
X_std[:, 1] = (X[:, 1] - X[:,1].mean()) / X[:,1].std()


ada = AdalineGD(n_iter = 15, eta = .01)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc = 'upper left')
plt.show()
plt.plot(range(1, len(ada.cost_)+ 1), ada.cost_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()



