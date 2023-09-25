import numpy as np
import os
import random
import tensorflow as tf
import time

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras

from transformer_main import create_model
from utils import *


# make more reproducible results, GPU does not allow full reproducibility
os.environ["PYTHONHASHSEED"] = "0"
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)

if __name__ == "__main__":
    for person in ["joel", "manuel", "pedro"]:
        for train_sessions, test_sessions in zip([["2","3","4"], ["1","3","4"], ["1","2","4"], ["1","2","3"]], [["1"],["2"],["3"],["4"]]):
            # read data
            x_train, y_train = read_dataset2(people=[person], sessions=train_sessions)
            x_test, y_test = read_dataset2(people=[person], sessions=test_sessions)

            input_shape = x_train.shape[1:]

            # data shuffling
            x_train, y_train = shuffle(x_train, y_train, random_state=0)

            # data splitting
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=1/4, random_state=0, stratify=y_train, shuffle=True)
            
            # model training and evaluation
            model = create_model(input_shape)

            callbacks = [keras.callbacks.EarlyStopping(patience=200, restore_best_weights=True)]

            start_time = time.time()
            results = model.fit(
                x_train,
                y_train,
                validation_data=(x_val,y_val),
                epochs=10000,
                batch_size=256,
                callbacks=callbacks,
            )
            t = time.time() - start_time
            print(t)


            l, a = model.evaluate(x_val, y_val, verbose=1)

            L, A = model.evaluate(x_test, y_test, verbose=1)

            # plots and save results
            plot_accuracy_comparison([results.history["sparse_categorical_accuracy"],
                                    results.history["val_sparse_categorical_accuracy"]],
                                    "Training/Validation Accuracy Comparison",
                                    ["Training Accuracy", "Validation Accuracy"],
                                    show=False, save_path = f"./results/transformer_{person}_{test_sessions[0]}_acc_comparison.svg")

            plot_loss_comparison([results.history["loss"], results.history["val_loss"]],
                                "Training/Validation Loss Comparison",
                                ["Training Loss", "Validation Loss"],
                                show=False, save_path = f"./results/transformer_{person}_{test_sessions[0]}_loss_comparison.svg")

            y_pred=np.argmax(model.predict(x_test), axis=-1)

            plot_confusion_matrix(y_test, y_pred, ["bottle", "cube", "phone", "screwdriver"],
                                show=False, save_path = f"./results/transformer_{person}_{test_sessions[0]}_conf_matrix.svg")

            write_results(results.history['sparse_categorical_accuracy'][-200], a, A,
                        results.history['loss'][-200], l, L,
                        classification_report(y_pred,y_test, digits=4),
                        training_time = t,
                        save_path = f"./results/transformer_{person}_{test_sessions[0]}_results.txt")
