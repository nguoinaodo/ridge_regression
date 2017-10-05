from model.Ridge import Ridge
from model.LinearRegression import LinearRegression
from sklearn.linear_model import Ridge as SKRidge, LinearRegression as SKLin
import numpy as np 
# Dim = 2
# X = [[1., 1], [2., 1], [3.,2]]
# y = [1., 3., 4.4]
# X_test = [[4.,4.], [2.3,2], [6.6,5]]

# Dim = 1
# X = [[1.], [2.], [3.], [4.5], [2.3]]
# y = [1., 3., 4.4, 4.5, 2.4]
# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183, 185, 187, 190, 193, 196, 200]]).T
# weight (kg)
y = np.array([[49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68, 70, 73, 78, 82, 86, 90]]).T

X_test = np.array([[148], [152], [155]])
y_test = np.array([49.3, 50.2, 52.7])
# from preprocessing.extract_features import \
# 		X_train, X_val, y_train, y_val, test_features

reg = Ridge(lamb=.1)
reg.fit(X, y)
# reg.fit(X_train, y_train)
reg.plot()
print reg.cost(X_test, y_test)
print reg.get_weights()
# print X, y, X_test, y_test

skreg = SKRidge(alpha=.1)
skreg.fit(X, y)
print skreg.intercept_
print skreg.coef_

# Linear
lin = LinearRegression()
lin.fit(X, y)
lin.plot()
print lin.cost(X_test, y_test)
print lin.get_weights()
# print X, y, X_test, y_test

sklin = SKLin()
sklin.fit(X, y)
print sklin.intercept_
print sklin.coef_
