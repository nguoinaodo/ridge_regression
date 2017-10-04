from model.Ridge import Ridge
import matplotlib.pyplot as plt 
import numpy as np 

from preprocessing.extract_features import \
		X_train, X_val_k, y_train, y_val_k, \
		X_test, y_test, \
		test_predict_features, kfolds, test_predict

lamdas = []
ks = range(3, 20)
# Number of folds
for k in ks:
	# Fold number k
	print "Number of folds: %d" % k
	lamdas_k = []
	for train_idx, val_idx in kfolds(k, X_train, y_train):
		features_train = X_train[train_idx]
		features_val = X_train[val_idx]
		target_train = y_train[train_idx]
		target_val = y_train[val_idx]

		lamdas_i = np.linspace(0, 500, 1000)
		costs_i = []
		for l in lamdas_i:
			reg = Ridge(lamb = l)
			reg.fit(features_train, target_train)
			cost = reg.cost(features_val, target_val)
			costs_i.append(cost)
		# Min cost
		min_i = np.argmin(costs_i)
		lamdas_k.append(lamdas_i[min_i])

	# Avg lamda with number of folds k
	lamda_k = np.average(lamdas_k)
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

print "Cost of model: %f" % cost_of_model
print "Coef of model:"
print reg.get_params()

# Predict
print "Predict:"
pred = reg.predict(test_predict_features)
print pred

# Output
out_matrix = np.concatenate((test_predict, pred), axis = 1)

# Write CSV
import csv
file = open('data/20142776-test-result.csv', 'w')
writer = csv.writer(file, delimiter=',')
for r in out_matrix:
	writer.writerow(r)