from energy_ml.models.other_models import non_DNN_regressor
from energy_ml.models.neural_network import DNN_regressor
def test_all_regressors(X_train, y_train, X_valid, y_valid, DNN_parameters = None, verbose = 0):
    if DNN_parameters is None:
        validation_results_DNN,  regressor_list_DNN = DNN_regressor(X_train, y_train, X_valid, y_valid, verbose=verbose)
    else:
        validation_results_DNN,  regressor_list_DNN = DNN_regressor(X_train, y_train, X_valid, y_valid,
                                        num_epochs = DNN_parameters['num_epochs'], dropout_rate = DNN_parameters['dropout_rate'],
                                        patience = DNN_parameters['patience'], batch_size = DNN_parameters['batch_size'],
                                        verbose = DNN_parameters['verbose'], loss = DNN_parameters['loss'],
                                        metrics = DNN_parameters['metrics'])

    validation_results_nonDNN, regressor_list_nonDNN = non_DNN_regressor(X_train, y_train, X_valid, y_valid)

    validation_results = validation_results_nonDNN | validation_results_DNN
    regressor_list = regressor_list_nonDNN | regressor_list_DNN
    return validation_results, regressor_list