from model.Ridge import Ridge
import matplotlib.pyplot as plt 
import numpy as np 

from preprocessing.extract_features import \
		training_features, training_target, \
		test_features, kfolds

# Test k
kfolds_avg_lambs = []
kfolds_avg_cost = []
min_cost = 9999999
K = 1
lamb_final = 1

for k in range(10, 15):
	# K-folds
	opt_lambs = []
	opt_costs = []
	for train_idx, val_idx in kfolds(k, training_features):
		X_train = training_features[train_idx]
		X_val = training_features[val_idx]
		y_train = training_target[train_idx]
		y_val = training_target[val_idx]

		# Validate parameter
		range1 = np.linspace(0.000001, 10, 10000)
		range2 = np.linspace(10, 1000, 1000)
		# range3 = np.linspace(1000, 8000000000, 10000)
		# lambs = np.concatenate((range1, range2, range3))
		lambs = np.concatenate((range1, range2))
		costs = []
		for l in lambs:
			reg = Ridge(lamb = l)
			reg.fit(X_train, y_train)
			cost = reg.cost(X_val, y_val)
			costs.append(cost)

		# Optimize
		min_i = np.argmin(costs)
		opt_lamb = lambs[min_i]
		opt_cost = costs[min_i]

		# Plot
		plt.plot(lambs, costs, 'r')
		plt.plot(opt_lamb, opt_cost, 'bo')
		plt.xlabel('lambda')
		plt.ylabel('cost')
		plt.show()
		 
		print 'Optimize lambda: %f' % opt_lamb
		opt_lambs.append(opt_lamb)
		opt_costs.append(opt_cost)

	lamb = np.average(opt_lambs[1:])
	print 'Average lambda: %f' % lamb
	cost = np.average(opt_costs[1:])
	print 'Average cost: %f' % cost
	kfolds_avg_lambs.append(lamb)
	kfolds_avg_cost.append(cost)
	if cost < min_cost:
		K = k
		lamb_final = lamb

# Kfolds result
plt.plot(range(20, 30, 3), kfolds_avg_cost, 'b')
plt.xlabel('k')
plt.ylabel('cost')
plt.show()

# Result
print 'K: %d' % K
print 'Final lambda: %f' % lamb_final
# 3182600

# 11.213714