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


if __name__ == "__main__":
    for person in ["joel", "manuel", "pedro"]:
        sum_accs = 0
        sum_losses = 0
        sum_times = 0
        sum_precisions = 0
        sum_recalls = 0
        sum_f1_scores = 0
        
        for _ in range(50):
            # read data
            x, y = read_dataset2(people=[person])

            input_shape = x.shape[1:]

            # data shuffling
            x, y = shuffle(x, y)

            # data splitting
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/5, stratify=y, shuffle=True)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=1/5, stratify=y_train, shuffle=True)

            random_indices = np.random.choice(len(y_test), size=732, replace=False)

            x_test = x_test[random_indices]
            y_test = y_test[random_indices]
            
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

            y_pred=np.argmax(model.predict(x_test), axis=-1)

            cr = classification_report(y_pred,y_test, digits=4)

            cr_last_line = cr.split("\n")[-2]
            precision = float(cr_last_line.split()[-4])
            recall = float(cr_last_line.split()[-3])
            f1_score = float(cr_last_line.split()[-2])

            sum_accs += A
            sum_losses += L
            sum_times += t
            sum_precisions += precision
            sum_recalls += recall
            sum_f1_scores += f1_score

            # plots and save results
            # plot_accuracy_comparison([results.history["sparse_categorical_accuracy"],
            #                         results.history["val_sparse_categorical_accuracy"]],
            #                         "Training/Validation Accuracy Comparison",
            #                         ["Training Accuracy", "Validation Accuracy"],
            #                         show=False, save_path = f"./results/transformer_{person}_session_{test_sessions[0]}_acc_comparison.svg")

            # plot_loss_comparison([results.history["loss"], results.history["val_loss"]],
            #                     "Training/Validation Loss Comparison",
            #                     ["Training Loss", "Validation Loss"],
            #                     show=False, save_path = f"./results/transformer_{person}_session_{test_sessions[0]}_loss_comparison.svg")

            # plot_confusion_matrix(y_test, y_pred, ["bottle", "cube", "phone", "screwdriver"],
            #                     show=False, save_path = f"./results/transformer_{person}_session_{test_sessions[0]}_conf_matrix.svg")

            # write_results(results.history['sparse_categorical_accuracy'][-200], a, A,
            #             results.history['loss'][-200], l, L,
            #             classification_report(y_pred,y_test, digits=4),
            #             training_time = t,
            #             save_path = f"./results/transformer_{person}_session_{test_sessions[0]}_results.txt")

        f = open(f"./results/transformer_intra_user_{person}_results.txt", "w")
        f.write(f"Average accuracy: {sum_accs/50}")
        f.write(f"Average loss: {sum_losses/50}")
        f.write(f"Average time: {sum_times/50}")
        f.write(f"Average precision: {sum_precisions/50}")
        f.write(f"Average recall: {sum_recalls/50}")
        f.write(f"Average f1-score: {sum_f1_scores/50}")
        f.close()