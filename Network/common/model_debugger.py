import tensorflow
from keras import Model, models
import numpy as np

from globals import MODEL_PATH


def extract_weights(model):
    all_weights = []
    for model_layer in model.layers:
        layer_weight_bytes = []
        layer_weights = model_layer.get_weights()
        # Extract weights
        for step in layer_weights[0]:
            for weight in step:
                # print(f"Value: {weight}, Hex: {half_to_hex(weight)}, Bytes:{weight_bytes}")
                # print("0b",''.join(format(x, '08b') for x in weight_bytes))
                layer_weight_bytes.append(weight)
        # Extract biases
        for bias in layer_weights[1]:
            layer_weight_bytes.append(bias)
        all_weights.append(layer_weight_bytes)
    return all_weights


def layer(output_length: int, input_layer: np.ndarray, model_weights) -> np.ndarray:
    """
    Copy of the network implementation in C#
    """
    layer_output = np.zeros(output_length + 1)
    for i in range(len(input_layer)):
        for j in range(output_length):
            weight_index = i * output_length + j
            weight = model_weights[weight_index]
            layer_output[j] += weight * input_layer[i]
    for i in range(output_length):
        layer_output[i] = np.tanh(layer_output[i]) if output_length == 1 else np.max([0, layer_output[i]])
    layer_output[output_length] = 1
    return layer_output


test_position = [
    4,  0,  3,  0,  6,  3,  2,  4,
    1,  1,  1,  0,  1,  1,  1,  1,
    2,  0,  0,  5,  0,  0,  0,  0,
    0,  0,  0,  1,  0,  0,  0,  0,
    0,  0,  0,  -1, -1, 0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
    -1, -1, -1, 0,  0,  -1, -1, -1,
    -4, -2, -3, -5, -6, -3, -2, -4,
]

trained_model: Model = models.load_model(MODEL_PATH)
print(f"Keras Model Prediction:\t\t{trained_model.predict(np.array([test_position]), verbose=False)[0][0]}")

weights = extract_weights(trained_model)
output_length_of_layers = [layer.output.shape[1] for layer in trained_model.layers]

output = layer(output_length_of_layers[0], np.array([*test_position, 1]), weights[0])
output = layer(output_length_of_layers[1], output, weights[1])
output = layer(output_length_of_layers[2], output, weights[2])
output = layer(output_length_of_layers[3], output, weights[3])

print(f"Manual Model Prediction:\t{output[0]}")
