from model.Ridge import Ridge
import matplotlib.pyplot as plt 
import numpy as np 

from preprocessing.extract_features import \
		X_train, X_val, y_train, y_val, test_features

lamb = 1
reg = Ridge(lamb)
reg.fit(X_train, y_train)
cost = reg.cost(X_val, y_val)

print 'Lambda = %f' % lamb
print 'Cost = %f' % cost

print reg.predict(test_features)