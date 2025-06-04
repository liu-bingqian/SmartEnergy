from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin
import numpy as np
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
def non_DNN_regressor(X_train, y_train, X_test, y_test, regressor_list = None):
    removed_regressors = [
        "TheilSenRegressor",
        "ARDRegression",
        "CCA",
        "IsotonicRegression",
        "StackingRegressor",
        "MultiOutputRegressor",
        "MultiTaskElasticNet",
        "MultiTaskElasticNetCV",
        "MultiTaskLasso",
        "MultiTaskLassoCV",
        "PLSCanonical",
        "PLSRegression",
        "RadiusNeighborsRegressor",
        "RegressorChain",
        "VotingRegressor",
        "GaussianProcessRegressor",
        "KernelRidge"
    ]

    if regressor_list is None:
        REGRESSORS = [
            est
            for est in all_estimators()
            if (issubclass(est[1], RegressorMixin) and (est[0] not in removed_regressors))
        ]
    else:
        REGRESSORS = [
            est
            for est in all_estimators()
            if (issubclass(est[1], RegressorMixin) and (est[0] not in removed_regressors) and (est[0] in regressor_list))
        ]

    ErrorResultsDict = {}
    TrainedModelDict = {}

    for model in REGRESSORS:
        start_time = time.time()
        model_name = model[0]

        print("Current Model Name: ", model_name)

        model_regressor = model[1]()

        try:
            model_regressor.fit(X_train, y_train)
        except ValueError:
            print("Current model is 1d model: ", model_name)
            continue

        y_pred = model_regressor.predict(X_test)

        # MSE
        mse = mean_squared_error(y_test, y_pred)

        # MAE
        mae = mean_absolute_error(y_test, y_pred)

        y_pred_train = model_regressor.predict(X_train)
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

        TrainedModelDict[model_name] = model_regressor

    return ErrorResultsDict, TrainedModelDict