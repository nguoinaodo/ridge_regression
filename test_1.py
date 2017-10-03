from model.LinearRegression import LinearRegression

# Dim = 2
# X = [[1., 1], [2., 1], [3.,2]]
# y = [1., 3., 4.4]
# X_test = [[4.,4.], [2.3,2], [6.6,5]]

# Dim = 1
# X = [[1.], [2.], [3.], [4.5], [2.3]]
# y = [1., 3., 4.4, 4.5, 2.4]
# X_test = [[4.], [2.3], [6.6]]

from preprocessing.extract_features import \
		X_train, X_val, y_train, y_val, test_features

reg = LinearRegression()
reg.fit(X_train, y_train)
reg.plot()
 
print reg.predict(X_val)
w = reg.get_weights()
print w 