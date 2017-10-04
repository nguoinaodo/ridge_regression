from model.Ridge import Ridge
import matplotlib.pyplot as plt 
import numpy as np 

from preprocessing.extract_features import \
		X_train, X_val, y_train, y_val, test_features

# Validate parameter
lambs = np.linspace(0, 1000, 10000)
costs = []
for l in lambs:
	reg = Ridge(lamb = l)
	reg.fit(X_train, y_train)
	cost = reg.cost(X_val, y_val)
	costs.append(cost)

# Optimize
min_i = np.argmin(costs)
lamb = lambs[min_i]
cost = costs[min_i]

# Plot
plt.plot(lambs, costs, 'r')
plt.plot(lamb, cost, 'bo')
plt.xlabel('lambda')
plt.ylabel('cost')
plt.show()
 
print 'Optimize lambda: %f' % lamb