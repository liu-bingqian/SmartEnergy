KFOLD_SPLITS = 5
DEFAULT_DNN_CONFIG = {
    'name': '5-hidden-layer',
    'activation': ['relu','relu', 'sigmoid'],
    'hidden_layers' : [1024, 64, 32],

}

hidden_layer_options = [
    [512, 128, 64],   # Reduce the first layer size, increase the middle layers
    [1024, 512, 128, 32],  # Add an extra middle layer
    [1024, 256, 128, 64, 32],  # More layers with gradual decrease in neurons
    [512, 256, 128, 64],  # Reduce the first layer size while maintaining depth
    [1024, 512, 256, 128, 64, 32],  # Increase depth with gradually decreasing neurons
    [1024, 512, 256, 128, 64],
    [1024, 512, 256],  # Balanced structure
    [512, 128, 64, 32, 16],  # Gradual decrease in neurons per layer
    [1024, 512, 128, 64],  # Moderately deep structure
    [256, 128, 64, 32],  # Smaller model, lower computational cost
    [1024, 256, 128, 32],  # Medium-sized model
    [512, 256, 128],  # Lightweight architecture
    [1024, 512, 256, 128, 64, 32, 16],  # Deep architecture but with smaller neuron count
]

DEFAULT_DNN_CONFIG_List = []

for i in range(len(hidden_layer_options)):
    DNN_Config_dict = dict()
    DNN_Config_dict['name'] = f'DeepNeuralNetwork_model_{i}'
    DNN_Config_dict['hidden_layers' ] = hidden_layer_options[i]
    DNN_Config_dict['activation'] = ['relu']*(len(hidden_layer_options[i])) + ['sigmoid']
    DEFAULT_DNN_CONFIG_List.append(DNN_Config_dict)
