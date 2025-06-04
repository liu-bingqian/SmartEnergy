import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
def scale_matrix_0_1(X, isByRow = True, Rescale_Type = 'normal', upper_bound = None, lower_bound = None, augmented_segment = 0.3):
    """
    Rescale entries of the matrix to interval [0,1].
    Max and min values are taken along columns.

    argument:
    X: NumPy array, shape (n_samples, n_features)

    return:
    scaled_X: matrix after rescaling
    """

    X = np.nan_to_num(X)
    X[X < 0] = 0

    if Rescale_Type == 'none':
        return 0, 0, X

    if upper_bound is None:
        if isByRow:
            X_max = np.max(X, axis=0)
        else:
            X_max = np.max(X)
    else:
        X_max = upper_bound

    if lower_bound is None:
        if isByRow:
            X_min = np.min(X, axis=0)
        else:
            X_min = np.min(X)
    else:
        X_min = lower_bound

    range_vector = X_max - X_min
    if type(range_vector) != np.float64 and type(range_vector) != np.int64:
        range_vector[range_vector <1e-5] = 1
    scaled_X = (X - X_min) / (range_vector+1e-9) #Nan may appear due to division by zero
    scaled_X = np.nan_to_num(np.float32(scaled_X)) #All nan values to 0

    if Rescale_Type == 'sqrt':
        scaled_X = np.where(scaled_X >= 0, np.sqrt(scaled_X), 0)
    elif Rescale_Type == 'log':
        scaled_X = np.where(scaled_X >= 0, np.log1p(scaled_X)/np.log(2), 0)

    augmented_segment = augmented_segment
    new_array = np.where(scaled_X  > 0, augmented_segment, 0)
    scaled_X = (scaled_X + new_array) / (1 + augmented_segment)

    return X_max, X_min, scaled_X

def inverse_scale_matrix_0_1(y, Parameter_Dict):
    """
    Rescale entries of the matrix to interval [0,1].
    Max and min values are taken along columns.

    argument:
    X: NumPy array, shape (n_samples, n_features)

    return:
    scaled_X: matrix after rescaling
    """
    y = np.nan_to_num(y)
    Rescale_Type = Parameter_Dict['Configuration_Rescale_Type']
    augmented_segment = Parameter_Dict['augmented_segment']

    y_max, y_min = Parameter_Dict[('Configuration', 'max')], Parameter_Dict[('Configuration', 'min')]

    scaled_y = y

    new_array = np.where(scaled_y  > 1e-5, augmented_segment, 0)
    scaled_y = scaled_y*(1 + augmented_segment) - new_array

    if Rescale_Type == 'sqrt':
        scaled_y = np.square(y)
    elif Rescale_Type == 'log':
        scaled_y = np.expm1(y*np.log(2))
    else:
        scaled_y = y

    range_vector = y_max - y_min

    if type(range_vector) != np.float64:
        range_vector[range_vector < 1e-5] = 1

    scaled_y = scaled_y*range_vector + y_min#Nan may appear due to division by zero
    scaled_y = np.nan_to_num(np.float32(scaled_y)) #All nan values to 0

    return scaled_y

def shuffle_dataset(X, y):
    '''
    Shuffle the dataset order
    '''
    Index_List = list(range(X.shape[0]))
    random.shuffle(Index_List)
    X_next = X[Index_List]
    y_next = y[Index_List]
    return X_next, y_next

def augment_typical_days(X, y, Parameter_Dict, augmented_number):
    """
    Augment dataset by permuting typical days in Temperature_Matrix, Irradiation_Matrix, and TypicalDay_Matrix.

    Parameters:
    - X: array, all dataset, obtained by np.concatenate([Building_Matrix, Temperature_Matrix, Irradiation_Matrix, TypicalDay_Matrix, MarketData_Matrix], axis=1)
    - Parameter_Dict: dict, it contains the size of each submatrix in Parameter_Dict['index_ranges']
    - augmented_number: int, total number of augmented samples to be generated.

    Returns:
    - Augmented dataset: (Original + Augmented, Feature_Dim)
    """

    augmented_number = int(augmented_number)
    num_original = X.shape[0]  # Number of samples in training dataset
    num_augments_needed = max(0, augmented_number)  # Number of new samples to generate
    X_augmented = []
    y_augmented = []

    for _ in range(num_augments_needed):
        # Randomly select a row index from original data
        idx = np.random.randint(0, num_original)

        Building_range = range(Parameter_Dict['index_ranges']['Building'][0],
                               Parameter_Dict['index_ranges']['Building'][1])
        Temperature_main_range = range(Parameter_Dict['index_ranges']['Temperature'][0],
                                       Parameter_Dict['index_ranges']['Temperature'][1] - 2)
        Temperature_extreme_range = range(Parameter_Dict['index_ranges']['Temperature'][1] - 2,
                                          Parameter_Dict['index_ranges']['Temperature'][1])
        Irradiation_main_range = range(Parameter_Dict['index_ranges']['Irradiation'][0],
                                       Parameter_Dict['index_ranges']['Irradiation'][1] - 2)
        Irradiation_extreme_range = range(Parameter_Dict['index_ranges']['Irradiation'][1] - 2,
                                          Parameter_Dict['index_ranges']['Irradiation'][1])
        TypicalDay_range = range(Parameter_Dict['index_ranges']['TypicalDay'][0],
                                 Parameter_Dict['index_ranges']['TypicalDay'][1])
        MarketData_range = range(Parameter_Dict['index_ranges']['MarketData'][0],
                                 Parameter_Dict['index_ranges']['MarketData'][1])

        # Extract selected row data
        Building_data = X[idx, Building_range].reshape(1, -1)
        Temperature_main_data = X[idx, Temperature_main_range].reshape(10, 24)
        Temperature_extreme_data = X[idx, Temperature_extreme_range].reshape(1, -1)
        Irradiation_main_data = X[idx][Irradiation_main_range].reshape(10, 24)
        Irradiation_extreme_data = X[idx][Irradiation_extreme_range].reshape(1, -1)
        TypicalDay_data = X[idx][TypicalDay_range].reshape(1, -1)
        MarketData_data = X[idx][MarketData_range].reshape(1, -1)

        Configuration_data = y[idx]

        # Generate a random permutation of 10 typical days
        permuted_indices = np.random.permutation(10)
        # Apply the permutation to the first 240 columns
        Temperature_main_data_permuted = Temperature_main_data[permuted_indices].reshape(1,
                                                                                         -1)  # Reshape back to (1, 240)
        Irradiation_main_data_permuted = Irradiation_main_data[permuted_indices].reshape(1,
                                                                                         -1)  # Reshape back to (1, 240)
        TypicalDay_data_permuted = TypicalDay_data[:, permuted_indices]  # (1, 10)

        # Concatenate all data for the new row
        new_row = np.concatenate(
            [Building_data, Temperature_main_data_permuted, Temperature_extreme_data, Irradiation_main_data_permuted,
             Irradiation_extreme_data, TypicalDay_data_permuted,
             MarketData_data], axis=1)

        X_augmented.append(new_row)
        y_augmented.append(Configuration_data)

    if num_augments_needed > 0:
        # Convert list to numpy array and concatenate with original data
        X_augmented_matrix = np.vstack(X_augmented)
        y_augmented_matrix = np.vstack(y_augmented)

        X_new = np.vstack([X, X_augmented_matrix])
        y_new = np.vstack([y, y_augmented_matrix])

    return X_new, y_new

def calculate_mae_list(y_pred, y_valid):
    mae_list = []
    for y_p, y_v in zip(y_pred, y_valid):
        y_p = np.array(y_p)
        y_v = np.array(y_v)
        mae = np.mean(np.abs(y_p - y_v))
        mae_list.append(mae)
    return mae_list

def plot_mae_vs_confidence(mae_list, confidence_rate, title = "MAE vs standard deviation", ylabel = "MAE on validation dataset"):
    plt.figure(figsize=(8, 5))
    plt.scatter(confidence_rate, mae_list, alpha=0.7)
    plt.xlabel("Confident score")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(bottom = 0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

