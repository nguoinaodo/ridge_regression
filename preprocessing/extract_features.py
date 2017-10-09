from read import train, test_predict
import numpy as np 
from sklearn.model_selection import train_test_split, \
		KFold, StratifiedKFold

# To array
training_data = np.array(train[1:]).astype(np.float)
training_features = training_data[:, :-1]
training_target = training_data[:, -1]
test_predict = np.array(test_predict)[:, :-1]
test_predict_features = test_predict.astype(np.float)

# Normalize
# train_means = np.mean(training_features, axis = 0)
# train_std = np.mean((training_features - train_means) ** 2, axis = 0) ** .5
# training_features = (training_features - train_means) / train_std
# test_predict_features = (test_predict.astype(np.float) - train_means) / train_std

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
		training_features, training_target,
		test_size = 0.25, random_state = 42)

# Validate k (K-folds)
X_train, X_val_k, y_train, y_val_k = train_test_split(X_train, y_train,
		test_size = 0.25, random_state = 42)

# K-folds
def kfolds(k, X, y):
	kf = KFold(n_splits = k)
	return kf.split(X, y)