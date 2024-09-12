import json
import random
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Add, Dense, Input
from tensorflow.keras.models import Model

# Load the trained model
model_path = 'h5 models\\trymodel.h5'
model = load_model(model_path)

def extract_model_info(model):
    model_info = {
        'layers': []
    }
    for layer in model.layers:
        layer_info = {
            'name': layer.name,
            'class_name': layer.__class__.__name__,
            'config': layer.get_config(),
            'weights': [w.tolist() for w in layer.get_weights()]
        }
        model_info['layers'].append(layer_info)
    return model_info

# Extract model information
model_info = extract_model_info(model)

# Save model information to a JSON file
with open('jsonfiles\\model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)

def random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def obfuscate_layer_names(model_info):
    existing_names = set()
    for layer in model_info['layers']:
        new_name = random_string()
        while new_name in existing_names:
            new_name = random_string()
        layer['name'] = new_name
        existing_names.add(new_name)
    return model_info

def encapsulate_parameters(model_info):
    for layer in model_info['layers']:
        layer['weights'] = [weights.tolist() if isinstance(weights, np.ndarray) else weights for weights in layer['weights']]
    return model_info

def add_shortcuts(model_info):
    # Only add shortcuts for layers that make sense (e.g., not adding Dense after Add)
    for i in range(len(model_info['layers']) - 1):
        if model_info['layers'][i]['class_name'] != 'Add' and model_info['layers'][i + 1]['class_name'] != 'Add':
            shortcut_layer = {
                'name': random_string(),
                'class_name': 'Add',
                'config': {},
                'weights': []
            }
            model_info['layers'].insert(i + 1, shortcut_layer)
    return model_info

def inject_extra_layers(model_info):
    # Avoid injecting a Dense layer if the previous layer was Dense or Add
    for i in range(len(model_info['layers']) - 1):
        if model_info['layers'][i]['class_name'] != 'Dense' and model_info['layers'][i]['class_name'] != 'Add':
            extra_layer = {
                'name': random_string(),
                'class_name': 'Dense',
                'config': {'units': random.randint(1, 100), 'activation': 'relu'},
                'weights': []
            }
            model_info['layers'].insert(i + 1, extra_layer)
    return model_info

# Apply obfuscation strategies
model_info = obfuscate_layer_names(model_info)
model_info = encapsulate_parameters(model_info)
model_info = add_shortcuts(model_info)
model_info = inject_extra_layers(model_info)

def ensure_unique_layer_names(model_info):
    unique_names = set()
    for layer in model_info['layers']:
        new_name = layer['name']
        while new_name in unique_names:
            new_name = random_string()
        layer['name'] = new_name
        unique_names.add(new_name)
    return model_info

def build_obfuscated_model(model_info):
    input_layer = Input(shape=model.input_shape[1:])
    x = input_layer
    layer_dict = {}

    for layer_info in model_info['layers']:
        LayerClass = getattr(tf.keras.layers, layer_info['class_name'])
        
        if layer_info['class_name'] == 'Add':
            previous_layer_name = list(layer_dict.keys())[-1]
            x = Add(name=layer_info['name'])([x, layer_dict[previous_layer_name]])
        else:
            # Create the layer instance with the modified config
            config = layer_info['config']
            config['name'] = layer_info['name']
            layer_instance = LayerClass(**config)
            x = layer_instance(x)
            layer_dict[layer_info['name']] = x

    obfuscated_model = Model(inputs=input_layer, outputs=x)
    return obfuscated_model

# Ensure all layer names are unique
model_info = ensure_unique_layer_names(model_info)

# Build the obfuscated model
obfuscated_model = build_obfuscated_model(model_info)

# Save the obfuscated model
obfuscated_model_path = 'h5 models\\obfuscated_model.h5'
obfuscated_model.save(obfuscated_model_path)
