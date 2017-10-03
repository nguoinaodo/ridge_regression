from read import train, test
import numpy as np 
from sklearn.model_selection import train_test_split, \
		KFold, StratifiedKFold

# To array
training_data = np.array(train[1:]).astype(np.float)
training_features = training_data[:, :-1]
training_target = training_data[:, -1]
test_features = np.array(test).astype(np.float)[:, :-1]

# Normalize
train_means = np.mean(training_features, axis = 0)
# print train_means
train_std = np.mean(training_features ** 2, axis = 0) ** .5
# print train_std
training_features = (training_features - train_means) / train_std
# print training_features
test_features = (test_features - train_means) / train_std

# Train validate split
X_train, X_val, y_train, y_val = train_test_split(
		training_features, training_target,
		test_size = 0.33, random_state = 42)


# K-folds
def kfolds(k, training_features):
	kf = KFold(n_splits = k)
	return kf.split(training_features, training_target)