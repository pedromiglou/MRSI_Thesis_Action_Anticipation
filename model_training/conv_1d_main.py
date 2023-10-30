import numpy as np
# import os
# import random
import tensorflow as tf

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras

from utils import *


N_CLASSES = 4


# create new model function 
def create_model(input_shape, dropout=0.5, learning_rate=0.001, kernel_size=3, num_conv_layers=2):
    # Create a `Sequential` model and add a Dense layer as the first layer.
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    for _ in range(num_conv_layers):
        model.add(tf.keras.layers.Conv1D(64, kernel_size, activation='relu'))
    #model.add(tf.keras.layers.MaxPooling1D(2,1))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(N_CLASSES, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["sparse_categorical_accuracy"]
    )
    return model


if __name__ == "__main__":
    # make more reproducible results, GPU does not allow full reproducibility
    # os.environ["PYTHONHASHSEED"] = "0"
    # random.seed(1234)
    # np.random.seed(1234)
    # tf.random.set_seed(1234)

    # read data
    x, y = read_dataset2()

    input_shape = x.shape[1:]

    # data shuffling
    x, y = shuffle(x, y, random_state=0)

    # data splitting
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=1/5, random_state=0, stratify=y)

    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=1/4, random_state=0, stratify=y_temp, shuffle=True)

    # model training and evaluation
    model = create_model(input_shape)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=200, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("results/cnn_model.h5",
            monitor='val_loss',  # Optional: Monitor a specific metric to save the best weights
            save_weights_only=False,  # Save the entire model
            save_best_only=True,  # Save only the best weights based on the monitored metric
            verbose=1  # Optional: Display messages when saving weights
        )
    ]

    results = model.fit(
        x_train,
        y_train,
        validation_data=(x_val,y_val),
        epochs=10000,
        batch_size=128,
        callbacks=callbacks,
    )

    l, a = model.evaluate(x_val, y_val, verbose=1)

    L, A = model.evaluate(x_test, y_test, verbose=1)

    # plots and save results
    plot_accuracy_comparison([results.history["sparse_categorical_accuracy"], results.history["val_sparse_categorical_accuracy"]],
                            "Training/Validation Accuracy Comparison",
                            ["Training Accuracy", "Validation Accuracy"],
                            show=False, save_path = "./results/cnn_acc_comparison.svg")

    plot_loss_comparison([results.history["loss"], results.history["val_loss"]],
                        "Training/Validation Loss Comparison",
                        ["Training Loss", "Validation Loss"],
                        show=False, save_path = "./results/cnn_loss_comparison.svg")

    y_pred=np.argmax(model.predict(x_test), axis=-1)

    plot_confusion_matrix(confusion_matrix(y_test, y_pred), ["bottle", "cube", "phone", "screw."],
                        show=False, save_path = "./results/cnn_conf_matrix.svg")
    
    write_results(results.history['sparse_categorical_accuracy'][-200], a, A,
                results.history['loss'][-200], l, L,
                classification_report(y_pred,y_test, digits=4),
                save_path = "./results/cnn_results.txt")
