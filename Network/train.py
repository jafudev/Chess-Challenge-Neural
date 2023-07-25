from typing import List
import pandas as pd
import numpy as np
import tensorflow
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.regularizers import l2
from main import format_model_to_c_sharp
import matplotlib.pyplot as plt
from globals import OUTPUT_CSHARP_MODEL_PATH, PLOTS_PATH, PROCESSED_DATA_CSV_PATH, TRAINED_MODEL_PATH, MODEL_PATH
from os import mkdir
from shutil import rmtree


TRAINING_SPLIT = 0.7
VALIDATION_SPLIT = 0.15
# TEST_SPLIT = 0.15 (1 - TRAINING_SPLIT - VALIDATION_SPLIT)


def extract_x_y(dataset):
    x = dataset.values[:, 0:64]
    y = dataset.values[:, 64]
    return x, y


def plot_history(history, folder: str):
    plot(history.history['loss'],
         history.history['val_loss'], 'loss', folder)
    # plot(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', index)


def plot(data: List, val_data: List, type: str, folder: str):
    plt.plot(data)
    plt.plot(val_data)
    plt.title(f'model {type}')
    plt.ylabel(type)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'{folder}{type}.png')
    plt.clf()


data = pd.read_csv(PROCESSED_DATA_CSV_PATH, header=None)

training, validation, test = \
            np.split(data.sample(frac=1, random_state=42), 
                    [int(TRAINING_SPLIT*len(data)), int((TRAINING_SPLIT + VALIDATION_SPLIT) * len(data))])



train_x, train_y = extract_x_y(training)
val_x, val_y = extract_x_y(validation)
test_x, test_y = extract_x_y(test)
    
# print(sum(train_y > 0.7) / len(train_y))

# tensorflow.keras.backend.set_floatx('float16')

model = Sequential()
model.add(Dense(200, input_shape=(64,),activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(200, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(200, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(1, activation='tanh', kernel_regularizer=l2(0.001)))
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001))
model.summary()

history = model.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(val_x, val_y))

try:
    mkdir(TRAINED_MODEL_PATH)
    mkdir(PLOTS_PATH)
except:
    rmtree(TRAINED_MODEL_PATH)
    mkdir(TRAINED_MODEL_PATH)
    mkdir(PLOTS_PATH)

plot_history(history, PLOTS_PATH)

model.save(MODEL_PATH)

c_sharp_model = format_model_to_c_sharp(model)
with open(OUTPUT_CSHARP_MODEL_PATH, 'w') as file:
    file.write(c_sharp_model)
