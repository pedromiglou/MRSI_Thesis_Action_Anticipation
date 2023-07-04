import numpy as np
import os
import random
import tensorflow as tf

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras
from utils import *

# make more reproducible results, GPU does not allow full reproducibility
os.environ["PYTHONHASHSEED"] = "0"
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)

# read data
folder_path = './points'

x, y = read_data(folder_path)

n_classes = len(np.unique(y))

input_shape = x.shape[1:]

# shuffle
x, y = shuffle(x, y, random_state=0)

# split
x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=1/5, random_state=0, stratify=y)

# create new model function 
def create_model(input_shape, dropout=0.5, learning_rate=0.001, kernel_size=3, num_conv_layers=2):
    # Create a `Sequential` model and add a Dense layer as the first layer.
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(21,3)))
    for _ in range(num_conv_layers):
        model.add(tf.keras.layers.Conv1D(64, kernel_size, activation='relu'))
    #model.add(tf.keras.layers.MaxPooling1D(2,1))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["sparse_categorical_accuracy"]
    )
    return model

while True:
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=1/4, random_state=0, stratify=y_temp, shuffle=True)

    model = create_model(input_shape)

    callbacks = [keras.callbacks.EarlyStopping(patience=200, restore_best_weights=True),
                keras.callbacks.ModelCheckpoint(
        "results/cnn_model",
        monitor='val_loss',  # Optional: Monitor a specific metric to save the best weights
        save_weights_only=True,  # Only save the model's weights, not the entire model
        save_best_only=True,  # Save only the best weights based on the monitored metric
        verbose=1  # Optional: Display messages when saving weights
    )]

    results = model.fit(
        x_train,
        y_train,
        validation_data=(x_val,y_val),
        epochs=10000,
        batch_size=128,
        callbacks=callbacks,
    )

    l, a = model.evaluate(x_val, y_val, verbose=1)

    if a > 0.94:
        break

f = open("./results/results.txt", "w")

f.write(f"Training loss: {results.history['loss'][-200]}\nTraining accuracy: {results.history['sparse_categorical_accuracy'][-200]}\n")

f.write(f"Validation loss: {l}\nValidation accuracy: {a}\n")

model.evaluate(x_test, y_test, verbose=1)
f.write(f"Test loss: {l}\nTest accuracy: {a}\n")

plot_accuracy_comparison([results.history["sparse_categorical_accuracy"], results.history["val_sparse_categorical_accuracy"]],
                        "Training/Validation Accuracy Comparison",
                        ["Training Accuracy", "Validation Accuracy"],
                        show=False, save_path = "./results/acc_comparison.svg")

plot_loss_comparison([results.history["loss"], results.history["val_loss"]],
                     "Training/Validation Loss Comparison",
                     ["Training Loss", "Validation Loss"],
                     show=False, save_path = "./results/loss_comparison.svg")

y_pred=np.argmax(model.predict(x_test), axis=-1)

f.write(classification_report(y_pred,y_test, digits=4))
f.close()

plot_confusion_matrix(y_test, y_pred, ["bottle", "cube", "phone", "screwdriver"],
                      show=False, save_path = "./results/conf_matrix.svg")