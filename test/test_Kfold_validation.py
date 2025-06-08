from notebooks.utils import *

from energy_ml.models.KFold_validation import KFold_validation_DNN
from test_data_reading import retrieve_data

X_path = '../input/REHO/REHO_train_X.csv'
y_path = '../input/REHO/REHO_train_y.csv'

X, y, Parameter_Dict = retrieve_data(X_path, y_path)
print("X shape:", X.shape)
print("y shape:", y.shape)

# Shuffle dataset order (optional)
X, y = shuffle_dataset(X, y)

# DNN structure
Model_Config = {
'name': "customized_DNN",
'L2_regularization' : 1e-4,
'activation': ['relu', 'relu', 'sigmoid'],
'hidden_layers' : [512, 256, 128]
}

results_dict = KFold_validation_DNN(X, y, model_list= [Model_Config])