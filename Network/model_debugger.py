from keras import Model, models
import numpy as np

from globals import MODEL_PATH


def extract_weights(model):
    all_weights = []
    for layer in model.layers:
        layer_weight_bytes = []
        layer_weights=layer.get_weights()
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


def Layer(outputLength: int, input: np.ndarray, weights: np.ndarray) -> np.ndarray:
    output = np.zeros(outputLength + 1)
    for i in range(len(input)):
        for j in range(outputLength):
            weightIndex = i * outputLength + j
            weight = weights[weightIndex]
            output[j] += weight * input[i]
    for i in range(outputLength):
        output[i] = np.tanh(output[i]) if outputLength == 1 else np.max([0, output[i]])
    output[outputLength] = 1
    return output


model: Model = models.load_model(MODEL_PATH)

weights = extract_weights(model)

# Bias is added in last element
input_manual = np.array([4, 0, 3, 5, 6, 3, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -4, -2, -3, -5, -6, -3, -2, -4, 1])
input_keras = np.array([[4, 0, 3, 5, 6, 3, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -4, -2, -3, -5, -6, -3, -2, -4]])

print(model.predict(input_keras))

input_manual = Layer(20, input_manual, weights[0])
input_manual = Layer(20, input_manual, weights[1])
input_manual = Layer(20, input_manual, weights[2])
input_manual = Layer(1, input_manual, weights[3])
print(input_manual)