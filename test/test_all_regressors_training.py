from notebooks.utils import *
import json
from energy_ml.models.test_all_regressors import test_all_regressors
from test_data_reading import retrieve_data

X_path = '../input/REHO/REHO_train_X.csv'
y_path = '../input/REHO/REHO_train_y.csv'

X, y, Parameter_Dict = retrieve_data(X_path, y_path)
print("X shape:", X.shape)
print("y shape:", y.shape)

# Shuffle dataset order (optional)
X, y = shuffle_dataset(X, y)

TrainNumber = int(X.shape[0] * 0.9)
X_train = X[:TrainNumber]
X_valid = X[TrainNumber:]

y_train = y[:TrainNumber]
y_valid = y[TrainNumber:]

# Data augmentation
'''
augmented_number = 3*X.shape[0]
X, y = augment_typical_days(X, y, Parameter_Dict, augmented_number)
'''

validation_results,  regressor_list = test_all_regressors(X_train, y_train, X_valid, y_valid, verbose=1)