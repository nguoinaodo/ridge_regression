import numpy as np 
import matplotlib.pyplot as plt 

class LinearRegression:
	def __init__(self):
		pass

	def fit(self, X, y):
		# Data
		self._Xtr = X = np.array(X) # NxD
		self._N = X.shape[0] 
		# Bias
		extends = np.ones((self._N, 1)) # Nx1
		X = np.concatenate((extends, X), axis = 1) # NxD'
		self._D = X.shape[1]
		self._ytr = y = np.array(y).reshape(self._N, 1) # Nx1
		# Weights
		self._w = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))\
				.reshape(self._D, 1) # D'x1
	
	def predict(self, X):
		X = np.array(X)
		N = X.shape[0]
		extends = np.ones((N, 1))
		X = np.concatenate((extends, X), axis = 1)
		pred = X.dot(self._w)
		return pred

	def get_weights(self):
		return self._w

	def plot(self):
		"""
		X: NxD
		y: Nx1
		"""
		if self._D == 2:
			# Points
			Xcol = self._Xtr[:, 0]
			ypred = self.predict(self._Xtr)
			plt.plot(Xcol, self._ytr, 'ro')
			# Line
			plt.plot(Xcol, ypred, 'b')
		# Show
		plt.show()