from model.Ridge import Ridge
import matplotlib.pyplot as plt 
import numpy as np 

from preprocessing.extract_features import \
		training_data, \
		X_train, X_val_k, y_train, y_val_k, \
		X_test, y_test, \
		test_predict_features, kfolds, test_predict


print training_data.shape
print X_train.shape
print y_train.shape
print X_val_k.shape
print y_val_k.shape
print X_test.shape
print y_test.shape
print test_predict_features.shape