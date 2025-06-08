from sklearn.model_selection import KFold
from energy_ml.models.neural_network import SequentialNeuralNetwork
import numpy as np
from energy_ml.models.other_models import non_DNN_regressor

def KFold_validation_DNN(X_train, y_train, model_list = []):

    if len(model_list) == 0:
        print('Test model list is empty')
        return

    n_splits = 5  # Value of K
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    Results_Dict = {}

    for Model_Config in model_list:
        fold_mae = []
        fold_mse = []
        print('Current Model Config:', Model_Config)
        i=1
        for train_index, valid_index in kf.split(X_train):
            print('Number of kFold', i)
            i+=1
            # 划分训练集和验证集
            X_train_kfold, X_valid = X_train[train_index], X_train[valid_index]
            y_train_kfold, y_valid = y_train[train_index], y_train[valid_index]
            model = SequentialNeuralNetwork(X_train_kfold, y_train_kfold, X_valid, y_valid, Model_Config=Model_Config)
            scores = model.evaluate(X_valid, y_valid, verbose=0)
            # 记录结果
            fold_mae.append(scores[1])  # MAE
            fold_mse.append(scores[2])  # MSE

            print(f"Fold MAE: {scores[1]:.4f}, Fold MSE: {scores[2]:.4f}")

        Model_Label = Model_Config['name'] + str(Model_Config['hidden_layers'])
        Results_Dict[Model_Label] = {'MAE' : np.mean(fold_mae), 'MSE' : np.mean(fold_mse)}
        print(f"Average MAE: {np.mean(fold_mae):.4f}, Std MAE: {np.std(fold_mae):.4f}")
        print(f"Average MSE: {np.mean(fold_mse):.4f}, Std MSE: {np.std(fold_mse):.4f}")
    return Results_Dict

def KFold_Validation_non_DNN(X_train, y_train, model_list = []):
    if len(model_list) == 0:
        print('Test model list is empty')
        return

    n_splits = 5  # K 的值
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    Results_Dict = {}

    for Model_Config in model_list:
        fold_mae = []
        fold_mape = []
        fold_mse = []
        print('Current Model Config:', Model_Config)
        i=1
        for train_index, valid_index in kf.split(X_train):
            print('Number of kFold', i)
            i+=1
            # 划分训练集和验证集
            X_train_kfold, X_valid = X_train[train_index], X_train[valid_index]
            y_train_kfold, y_valid = y_train[train_index], y_train[valid_index]
            ErrorResultsDict, TrainedModelDict = non_DNN_regressor(X_train_kfold, y_train_kfold, X_valid, y_valid, regressor_list = [Model_Config])
            # 记录结果
            fold_mae.append(ErrorResultsDict[Model_Config]['valid_MAE'])  # MAE
            fold_mse.append(ErrorResultsDict[Model_Config]['valid_MAPE'])  # MAPE
            fold_mape.append(ErrorResultsDict[Model_Config]['valid_MSE'])  # MAPE

            print(f"Fold MAE: {ErrorResultsDict[Model_Config]['valid_MAE']:.4f}, Fold MSE: {ErrorResultsDict[Model_Config]['valid_MSE']:.4f}, Fold MAPE: {ErrorResultsDict[Model_Config]['valid_MAPE']:.2f}%")

        Model_Label = Model_Config
        Results_Dict[Model_Label] = {'MAE' : np.mean(fold_mae), 'MSE' : np.mean(fold_mse), 'MAPE' : np.mean(fold_mape)}
        print(f"Average MAE: {np.mean(fold_mae):.4f}, Std MAE: {np.std(fold_mae):.4f}")
        print(f"Average MSE: {np.mean(fold_mse):.4f}, Std MSE: {np.std(fold_mse):.4f}")
        print(f"Average MAPE: {np.mean(fold_mape):.2f}%, Std MAPE: {np.std(fold_mape):.2f}%")
    return Results_Dict


