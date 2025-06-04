import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Input, Flatten, concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from keras import regularizers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import joblib
import pickle
from energy_ml.Config import *

def DNN_regressor(X_train, y_train, X_valid, y_valid, num_epochs = 5000, dropout_rate = 0, patience = 50, batch_size = 32,
                            verbose = 0, loss = 'mae', metrics = ['mae', 'mse'], regressor_list = None):
    df_list = []
    DNN_list = {}
    ErrorResultsDict = {}
    TrainedModelDict = {}

    DNN_List = []
    if regressor_list == None:
        DNN_List = DEFAULT_DNN_CONFIG_List
    else:
        for DNN_num in regressor_list:
            DNN_List.append(DEFAULT_DNN_CONFIG_List[DNN_num])

    for DNN_Config in DNN_List:
        start_time = time.time()

        model_name = DNN_Config['name']

        print("Current model name: ", model_name)
        print("Current model structure: ", DNN_Config['hidden_layers'])

        DL_model = SequentialNeuralNetwork(X_train, y_train, X_valid, y_valid, dropout_rate=0.1,
                                           Model_Config=DNN_Config, num_epochs=num_epochs, batch_size=batch_size, patience=patience,
                                           verbose=verbose)

        y_pred = DL_model.predict(X_valid)
        mae = mean_absolute_error(y_valid, y_pred)
        mse = mean_squared_error(y_valid, y_pred)

        y_pred_train = DL_model.predict(X_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mse_train = mean_squared_error(y_train, y_pred_train)

        training_time = time.time() - start_time
        ErrorResultsDict[model_name]  = {
            'valid_MAE': mae,
            'valid_MSE': mse,
            'train_MAE': mae_train,
            'train_MSE': mse_train,
            'training_time': training_time
        }

        TrainedModelDict[model_name] = DL_model

    return ErrorResultsDict, TrainedModelDict

def SequentialNeuralNetwork(X_train, y_train, X_valid, y_valid, Model_Config = None, num_epochs = 5000, dropout_rate = 0, patience = 50, batch_size = 32,
                            verbose = 0, loss = 'mae', metrics = ['mae', 'mse']):
    """
      Builds, compiles, and trains a customizable fully-connected feedforward neural network
      using Keras Sequential API.

      Parameters:
      ----------
      X_train : ndarray
          Training feature data.
      y_train : ndarray
          Training target data.
      X_valid : ndarray
          Validation/test feature data.
      y_valid : ndarray
          Validation/test target data.
      Model_Config : dict
          Dictionary specifying 'hidden_layers' (list of int) and 'activation' (str or list of str).
          Optionally includes 'L2_regularization' (float).
      num_epochs : int, optional
          Maximum number of training epochs (default is 5000).
      dropout_rate : float, optional
          Dropout rate between layers (default is 0).
      patience : int, optional
          Early stopping patience (default is 50).
      batch_size : int, optional
          Batch size for training (default is 32).
      verbose : int, optional
          Verbosity level for model training (0 = silent, 1 = progress bar, 2 = one line per epoch).
      loss : str, optional
          Loss function (default is 'mae').
      metrics : list, optional
          List of evaluation metrics (default is ['mae', 'mse']).

      Returns:
      -------
      model : keras.Sequential
          Trained Keras model with the best weights (according to early stopping).
      """

    # ==== Read model configurations ====
    if Model_Config is None:
        return

    X_Features = X_train.shape[1]
    try:
        y_Features = y_train.shape[1]
    except IndexError:
        y_Features = 1

    if 'L2_regularization' in Model_Config.keys():
        l2_lambda = Model_Config['L2_regularization']
    else:
        l2_lambda = 0

    activation = Model_Config['activation']
    hidden_layers = Model_Config['hidden_layers']

    #activation can be either a list or a string
    #If it's string, transform it into a list of the same length of hidden layers
    if type(activation) == str:
        activation = [activation]*(len(hidden_layers)+2)

    hidden_layer_list = [Dense(hidden_layers[0], activation=activation[0], input_shape=(X_Features,), kernel_regularizer=regularizers.l2(l2_lambda)),
                         Dropout(dropout_rate)]

    for num in range(len(hidden_layers)-1):
        hidden_layer_list.append(
            Dense(hidden_layers[num+1], activation=activation[num+1], kernel_regularizer=regularizers.l2(l2_lambda))
        )
        hidden_layer_list.append(
            Dropout(dropout_rate)
        )
    hidden_layer_list.append(
        Dense(y_Features, activation=activation[-1], kernel_regularizer=regularizers.l2(l2_lambda))
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    # ==== Model setup ====
    model = Sequential(hidden_layer_list)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=loss, metrics= metrics)

    # ==== Model training ====
    history = model.fit(X_train, y_train,
                        validation_data=(X_valid, y_valid),
                        epochs=num_epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping],
                        verbose=verbose)

    return model
