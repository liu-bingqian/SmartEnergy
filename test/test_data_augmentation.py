import numpy as np

from notebooks.utils import *
import json
from energy_ml.models.neural_network import DNN_regressor
from energy_ml.models.confidence_rate import confidence_rate

from test_data_reading import retrieve_data
from sklearn.metrics import mean_absolute_error, r2_score

X_path = '../input/REHO/REHO_train_X.csv'
y_path = '../input/REHO/REHO_train_y.csv'

X, y, Parameter_Dict = retrieve_data(X_path, y_path)
print("X shape:", X.shape)
print("y shape:", y.shape)

# Shuffle dataset order (optional)
X, y = shuffle_dataset(X, y)

TrainNumber = int(X.shape[0] * 0.9)
X_train_initial = X[:TrainNumber]
X_valid = X[TrainNumber:]

y_train_initial = y[:TrainNumber]
y_valid = y[TrainNumber:]


#Read test data

X_path_test = '../input/REHO/REHO_test_X.csv'
y_path_test = '../input/REHO/REHO_test_y.csv'

X_test, y_test, _ = retrieve_data(X_path_test, y_path_test, Parameter_Dict_predifined=Parameter_Dict)

for data_augment_rate in np.linspace(1, 10, 9):
    # Data augmentation
    augmented_number = 3*X.shape[0]
    X_train, y_train = augment_typical_days(X_train_initial , y_train_initial , Parameter_Dict, augmented_number)
    validation_results,  regressor_list = DNN_regressor(X_train, y_train, X_valid, y_valid, regressor_list = [11],verbose=1, batch_size= 12800)
    model = regressor_list['DeepNeuralNetwork_model_11']

    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    confidence_rate_vector = confidence_rate(model, X_test)

    print("Current Data Augmentation rate: ", data_augment_rate)
    print("mean MAE:", mae)
    print("mean R2:", r2)
    print("mean Confidence:", np.mean(confidence_rate_vector))
