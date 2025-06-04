from notebooks.utils import *
import json

def retrieve_data(X_path, y_path, Parameter_Dict_predifined = None):
    X_df = pd.read_csv(X_path)
    y_df = pd.read_csv(y_path)

    with open("../input/REHO/feature_slices.json", "r") as f:
        feature_slices = json.load(f)

    feature_slices = {k: tuple(v) for k, v in feature_slices.items()} # This defines the blocks of features in the input matrix X

    # Split X into blocks
    X_parts = {}
    for name, (start, end) in feature_slices.items():
        X_parts[name] = X_df.iloc[:, start:end].to_numpy()
    y_matrix = y_df.to_numpy()

    if Parameter_Dict_predifined is None:
        # Define parameter set
        Parameter_Dict = {}

        # Add feature slices into parameter set
        Parameter_Dict['index_ranges'] = feature_slices

        # Add configuration labels
        Parameter_Dict['Configuration_Labels'] = y_df.columns.tolist()
    else:
        Parameter_Dict = Parameter_Dict_predifined

    if Parameter_Dict_predifined is None:
        # Rescaling X_df. Temperature, irradiation and typical days are scale by the whole block
        Parameter_Dict[('Building', 'max')], Parameter_Dict[('Building', 'min')], Building_Matrix = scale_matrix_0_1(X_parts['Building'])
        Parameter_Dict[('Temperature', 'max')], Parameter_Dict[('Temperature', 'min')], Temperature_Matrix = scale_matrix_0_1(X_parts['Temperature'], isByRow=False)
        Parameter_Dict[('Irradiation', 'max')], Parameter_Dict[('Irradiation', 'min')], Irradiation_Matrix = scale_matrix_0_1(X_parts['Irradiation'], isByRow=False)
        Parameter_Dict[('TypicalDay', 'max')], Parameter_Dict[('TypicalDay', 'min')], TypicalDay_Matrix = scale_matrix_0_1(X_parts['TypicalDay'], isByRow=False)
        Parameter_Dict[('MarketData', 'max')], Parameter_Dict[('MarketData', 'min')], MarketData_Matrix = scale_matrix_0_1(X_parts['MarketData'])
    else:
        _, _, Building_Matrix = scale_matrix_0_1(
            X_parts['Building'],
            upper_bound=Parameter_Dict[('Building', 'max')],
            lower_bound=Parameter_Dict[('Building', 'min')]
        )

        _, _, Temperature_Matrix = scale_matrix_0_1(
            X_parts['Temperature'],
            isByRow=False,
            upper_bound=Parameter_Dict[('Temperature', 'max')],
            lower_bound=Parameter_Dict[('Temperature', 'min')]
        )

        _, _, Irradiation_Matrix = scale_matrix_0_1(
            X_parts['Irradiation'],
            isByRow=False,
            upper_bound=Parameter_Dict[('Irradiation', 'max')],
            lower_bound=Parameter_Dict[('Irradiation', 'min')]
        )

        _, _, TypicalDay_Matrix = scale_matrix_0_1(
            X_parts['TypicalDay'],
            isByRow=False,
            upper_bound=Parameter_Dict[('TypicalDay', 'max')],
            lower_bound=Parameter_Dict[('TypicalDay', 'min')]
        )

        _, _, MarketData_Matrix = scale_matrix_0_1(
            X_parts['MarketData'],
            upper_bound=Parameter_Dict[('MarketData', 'max')],
            lower_bound=Parameter_Dict[('MarketData', 'min')]
        )

    # Concatenate to get X
    X = np.concatenate([Building_Matrix, Temperature_Matrix, Irradiation_Matrix, TypicalDay_Matrix, MarketData_Matrix], axis=1)

    # Rescale y_df. The lower bound is set to 0 to avoid nan value when choosing sqrt or log scaler
    if Parameter_Dict_predifined is None:
        Rescale_Type = 'normal'
        augmented_segment = 0.3

        Parameter_Dict['Configuration_Rescale_Type'] = Rescale_Type
        Parameter_Dict['augmented_segment'] = augmented_segment

        Parameter_Dict[('Configuration', 'max')], Parameter_Dict[('Configuration', 'min')], y= scale_matrix_0_1(y_df, Rescale_Type= Rescale_Type, lower_bound= 0, augmented_segment = augmented_segment)
    else:
        _, _, y = scale_matrix_0_1(y_df, Rescale_Type=Parameter_Dict['Configuration_Rescale_Type'], upper_bound = Parameter_Dict[('Configuration', 'max')], lower_bound=0,
                                        augmented_segment=Parameter_Dict['augmented_segment'])

    return X, y, Parameter_Dict

if __name__ == '__main__':

    X_path = '../input/REHO/REHO_train_X.csv'
    y_path = '../input/REHO/REHO_train_y.csv'

    X, y, Parameter_Dict = retrieve_data(X_path, y_path)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_path_test = '../input/REHO/REHO_test_X.csv'
    y_path_test = '../input/REHO/REHO_test_y.csv'

    X_test, y_test, _ = retrieve_data(X_path_test, y_path_test, Parameter_Dict_predifined= Parameter_Dict)
    print("X shape:", X_test.shape)
    print("y shape:", y_test.shape)
