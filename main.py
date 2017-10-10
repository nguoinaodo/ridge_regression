from model.Ridge import Ridge
import matplotlib.pyplot as plt 
import numpy as np 

from preprocessing.extract_features import \
		X_train, X_val_k, y_train, y_val_k, \
		X_test, y_test, \
		test_predict_features, kfolds, test_predict

limit_lamda = [1, 10, 100, 1000, 10000]
# For other limits of lamda
model_lamda = 0
model_cost = 99999999999
model_coef = np.zeros(X_train.shape[0] + 1)
for lim in limit_lamda:
	lamdas = []
	ks = range(3, 20)
	# Number of folds
	for k in ks:
		# Fold number k
		print "Number of folds: %d" % k
		lamdas_k = np.linspace(0, lim, 1000)
		costs_k = []
		for l in lamdas_k:
			reg = Ridge(lamb = l)
			costs_l = []
			for train_idx, val_idx in kfolds(k, X_train, y_train):
				features_train = X_train[train_idx]
				features_val = X_train[val_idx]
				target_train = y_train[train_idx]
				target_val = y_train[val_idx]

				reg.fit(features_train, target_train)
				cost = reg.cost(features_val, target_val)
				costs_l.append(cost)
			cost_l = np.average(costs_l)
			costs_k.append(cost_l)	
		# Lamda with min average cost
		min_id = np.argmin(costs_k)
		lamda_k = lamdas_k[min_id]
		lamdas.append(lamda_k)
		print "lamda_k = %f" % lamda_k

	# Validate k
	costs = []
	for i in range(len(ks)):
		reg = Ridge(lamb = lamdas[i])
		reg.fit(X_train, y_train)
		cost = reg.cost(X_val_k, y_val_k)
		costs.append(cost)

	min_k = np.argmin(costs)
	# Optimize k and lamda
	k = ks[min_k]
	lamda = lamdas[min_k]

	# Plot
	plt.plot(ks, costs, 'r')
	plt.plot(k, costs[min_k], 'bo')
	plt.xlabel('k')
	plt.ylabel('cost')
	plt.show()
			 
	print "Final lamda: %f with number of folds = %d" % (lamda, k)

	# Score model
	X = np.concatenate((X_train, X_val_k), axis = 0)
	y = np.concatenate((y_train, y_val_k))
	reg = Ridge(lamb = lamda)
	reg.fit(X, y)
	cost_of_model = reg.cost(X_test, y_test)

	print "Cost of model (with limit of lamda = %d): %f" % (lim, cost_of_model)

	if cost_of_model < model_cost:
		model_lamda = lamda
		model_coef = reg.get_weights()
		model_pred = reg.predict(test_predict_features)
		model_cost = cost_of_model

print "Model lamda: %f" % model_lamda
print "Model cost: %f" % model_cost
print "Model coef:"
print model_coef
# Predict
print "Predict:"
print model_pred


# Output
out_matrix = np.concatenate((test_predict, model_pred), axis = 1)
# Write CSV
import csv
file = open('data/20142776-test-result-std.csv', 'w')
writer = csv.writer(file, delimiter=',')
for r in out_matrix:
	writer.writerow(r)