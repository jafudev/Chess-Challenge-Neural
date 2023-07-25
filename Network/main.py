import struct
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import tensorflow
import random as python_random


def half_to_hex(h) -> str:
    return hex(struct.unpack('<H', struct.pack('<e', h))[0])


def half_to_bytes(h) -> bytes:
    return struct.pack('<e', h)


def bytes_to_ulong(b) -> int:
    return struct.unpack('<Q', b)[0]


def extract_weights_in_bytes(model):
    all_weight_bytes = []
    for layer in model.layers:
        layer_weight_bytes = []
        layer_weights=layer.get_weights()
        # Extract weights
        for step in layer_weights[0]:
            for weight in step:
                weight_bytes = half_to_bytes(weight.astype('float16'))
                # print(f"Value: {weight}, Hex: {half_to_hex(weight)}, Bytes:{weight_bytes}")
                # print("0b",''.join(format(x, '08b') for x in weight_bytes))
                layer_weight_bytes.append(weight_bytes)
        # Extract biases
        for bias in layer_weights[1]:
            bias_bytes = half_to_bytes(bias)
            layer_weight_bytes.append(bias_bytes)
        all_weight_bytes.append(layer_weight_bytes)
    return all_weight_bytes

def group_weight_bytes(all_weight_bytes):
    grouped_weight_bytes = []
    for layer in all_weight_bytes:
        position_index = 0
        current_weight_bytes = b''
        layer_group_bytes = []
        # Group 4 weight bytes into one 8 byte object
        for weight_byte in layer:
            if position_index >= 4:
                layer_group_bytes.append(current_weight_bytes)
                position_index = 0
                current_weight_bytes = b''
            current_weight_bytes += weight_byte
            position_index += 1
        # Include excess weights
        if len(current_weight_bytes) > 0:
            byte_size = len(current_weight_bytes)
            padding = b'\x00' * (8 - byte_size)
            excess_weight_bytes = current_weight_bytes + padding
            layer_group_bytes.append(excess_weight_bytes)
        grouped_weight_bytes.append(layer_group_bytes)
    return grouped_weight_bytes


def encode_weight_bytes(grouped_weight_bytes):
    encoded_weights = []
    for layer in grouped_weight_bytes:
        encoded_layer_weights = []
        # Convert grouped 8 byte objects to ulong
        for weight_bytes in layer:
            encoded_weight = bytes_to_ulong(weight_bytes)
            encoded_layer_weights.append(encoded_weight)
        encoded_weights.append(encoded_layer_weights)
    return encoded_weights


def format_encoded_weights_to_c_sharp_array(encoded_weights):
    array_str = f'new ulong[{len(encoded_weights)}] {{'
    for long in encoded_weights:
        array_str += f'{long}, '
    array_str = array_str[:-2]
    array_str += ' }'
    return array_str


def format_model_to_c_sharp(model):
    all_weight_bytes = extract_weights_in_bytes(model)
    grouped_weight_bytes = group_weight_bytes(all_weight_bytes)
    encoded_weight_bytes = encode_weight_bytes(grouped_weight_bytes)

    model_str = ''
    for model_layer, encoded_layer in zip(model.layers, encoded_weight_bytes):
        model_str += f'input = Layer({model_layer.output_shape[1]}, input, {format_encoded_weights_to_c_sharp_array(encoded_layer)});\n'
    return model_str


def test_eval(model):
    pieces = [4, 0, 3, 5, 6, 3, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -4, -2, -3, -5, -6, -3, -2, -4]
    test_input = np.array([pieces])
    eval = model.predict(test_input)
    print(eval)


def fix_seeds(seed):
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)
    python_random.seed(seed)

