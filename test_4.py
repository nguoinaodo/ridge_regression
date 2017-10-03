from model.Ridge import Ridge
import matplotlib.pyplot as plt 
import numpy as np 

from preprocessing.extract_features import \
		X_train, X_val, y_train, y_val

# lambs = np.linspace(0.0001, 20000, 100000)
lambs = [7.220671, 11.213714, 1,2,3,4,5.847937,5.865847,6.809718,7.045694]
costs = []
for l in lambs:
	reg = Ridge(lamb = l)
	reg.fit(X_train, y_train)
	cost = reg.cost(X_val, y_val)
	costs.append(cost)

min_i = np.argmin(costs)
opt_lamb = lambs[min_i]
opt_cost = costs[min_i]
print "Optimize lambda: %f" % opt_lamb

# Plot
plt.plot(lambs, costs, 'ro')
plt.plot(opt_lamb, opt_cost, 'bo')
plt.xlabel('lambda')
plt.ylabel('cost')
plt.show()
		 

