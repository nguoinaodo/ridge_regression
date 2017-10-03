from model.Ridge import Ridge
import matplotlib.pyplot as plt 
import numpy as np 

from preprocessing.extract_features import \
		training_features, training_target, \
		test_features

reg = Ridge(lamb = 3182600)
reg.fit(training_features, training_target)
pred = reg.predict(test_features)

print pred