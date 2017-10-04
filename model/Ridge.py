import numpy as np 
import matplotlib.pyplot as plt 

class Ridge:
	def __init__(self, lamb):
		self._lamb = lamb # Regularization

	def fit(self, X, y):
		# Data
		self._Xtr = X = np.array(X)
		self._N = X.shape[0]
		# Extend
		extend = np.ones((self._N, 1))
		X = np.concatenate((extend, X), axis = 1)
		self._D = X.shape[1]
		self._ytr = y = np.array(y).reshape(self._N, 1)
		# Weights
		self._w = np.linalg.pinv(X.T.dot(X) + \
				self._lamb * np.eye(self._D)).dot(X.T.dot(y))
	
	def predict(self, X):	
		# Extend
		X = np.array(X)
		N = X.shape[0]
		extend = np.ones((N, 1))
		X = np.concatenate((extend, X), axis = 1)
		# Predict
		pred = X.dot(self._w)
		return pred

	def get_params(self):
		return self._w

	# RSS
	def cost(self, X, y):
		pred = self.predict(X)
		N = X.shape[0]
		cost = (np.linalg.norm(y - pred) ** 2) / N
		return cost

	# GCVE error
	def gcve(self, X, y):
		pred = self.predict(X)
		j = X.shape[0]
		n_CV = self._N
		df = np.trace(X.dot(np.linalg.pinv(X.T.dot(X) + self._lamb * np.eye(self._D - 1))).dot(X.T))
		gcve = 1. / j * np.linalg.norm(y - pred) ** 2 / (n_CV - df)

		return gcve
