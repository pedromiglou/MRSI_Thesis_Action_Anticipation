import numpy as np
import os
import random
import tensorflow as tf

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras
from utils import *

from transformer_main import create_model


# make more reproducible results, GPU does not allow full reproducibility
os.environ["PYTHONHASHSEED"] = "0"
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)


if __name__ == "__main__":
    PEOPLE = ["pedro"]

    # read data
    folder_path = './points'

    x, y = read_data(folder_path, people=PEOPLE)

    input_shape = x.shape[1:]

    # data shuffling
    x, y = shuffle(x, y, random_state=0)

    # data splitting
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=1/5, random_state=0, stratify=y)

    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=1/4, random_state=0, stratify=y_temp, shuffle=True)

    # model training and evaluation
    model = create_model(input_shape)

    callbacks = [keras.callbacks.EarlyStopping(patience=200, restore_best_weights=True)]

    results = model.fit(
        x_train,
        y_train,
        validation_data=(x_val,y_val),
        epochs=10000,
        batch_size=256,
        callbacks=callbacks,
    )

    l, a = model.evaluate(x_val, y_val, verbose=1)

    L, A = model.evaluate(x_test, y_test, verbose=1)

    # plots and save results
    plot_accuracy_comparison([results.history["sparse_categorical_accuracy"],
                            results.history["val_sparse_categorical_accuracy"]],
                            "Training/Validation Accuracy Comparison",
                            ["Training Accuracy", "Validation Accuracy"],
                            show=False, save_path = f"./results/transformer_{PEOPLE[0]}_acc_comparison.svg")

    plot_loss_comparison([results.history["loss"], results.history["val_loss"]],
                        "Training/Validation Loss Comparison",
                        ["Training Loss", "Validation Loss"],
                        show=False, save_path = f"./results/transformer_{PEOPLE[0]}_loss_comparison.svg")

    y_pred=np.argmax(model.predict(x_test), axis=-1)

    plot_confusion_matrix(y_test, y_pred, ["bottle", "cube", "phone", "screwdriver"],
                        show=False, save_path = f"./results/transformer_{PEOPLE[0]}_conf_matrix.svg")

    write_results(results.history['sparse_categorical_accuracy'][-200], a, A,
                results.history['loss'][-200], l, L,
                classification_report(y_pred,y_test, digits=4),
                save_path = f"./results/transformer_{PEOPLE[0]}_results.txt")
