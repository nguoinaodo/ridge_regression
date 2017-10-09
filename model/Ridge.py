import numpy as np 
import matplotlib.pyplot as plt 

class Ridge:
	def __init__(self, lamb):
		self._lamb = lamb # Regularization

	def fit(self, X, y):
		# Data
		self._Xtr = X = np.array(X) 
		# print X.shape
		self._N = X.shape[0]
		# Extend
		extend = np.ones((self._N, 1))
		X = np.concatenate((extend, X), axis = 1)
		# print X.shape
		self._D = X.shape[1]
		self._ytr = y = np.array(y).reshape(self._N, 1)
		# print y.shape
		# Weights
		self._w = np.linalg.inv(X.T.dot(X) + self._lamb * np.eye(self._D)).dot(X.T.dot(y))
	
	def predict(self, X):	
		# Extend
		X = np.array(X)
		N = X.shape[0]
		extend = np.ones((N, 1))
		X = np.concatenate((extend, X), axis = 1)
		# Predict
		pred = X.dot(self._w)
		return pred

	def get_weights(self):
		return self._w

	# RSS/N
	def cost(self, X, y):
		N = X.shape[0]
		y = y.reshape(N)
		pred = self.predict(X).reshape(N)
		cost = (np.linalg.norm(y - pred) ** 2) / N
		return cost

	# GCVE error
	def gcve(self, X, y):
		j = X.shape[0]
		y = y.reshape(j)
		pred = self.predict(X).reshape(j)
		n_CV = self._N
		df = np.trace(X.dot(np.linalg.inv(X.T.dot(X) + self._lamb * np.eye(self._D - 1))).dot(X.T))
		gcve = 1. / j * np.linalg.norm(y - pred) ** 2 / (n_CV - df)

		return gcve

	def plot(self):
		"""
		X: NxD
		y: Nx1
		"""
		if self._D == 2:
			# Points
			Xcol = self._Xtr[:, 0]
			xmax = max(Xcol)
			ypred = self.predict(self._Xtr)
			ymax = max(self._ytr)
			plt.plot(Xcol, self._ytr, 'ro')
			plt.xlim(0, xmax + 5)
			plt.ylim(0, ymax + 5)
			# Line
			plt.plot(Xcol, ypred, 'b')
		# Show
		plt.show()	