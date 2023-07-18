import numpy as np
import os
import random
import tensorflow as tf

from sklearn.model_selection import KFold, train_test_split
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
    # read data
    x, y = read_dataset2()

    input_shape = x.shape[1:]

    # data shuffling
    x, y = shuffle(x, y, random_state=0)

    # data splitting
    x, x_test, y, y_test = train_test_split(x, y, test_size=1/5, random_state=0, stratify=y)

    hyperparameters = {
        'learning_rate': [0.01, 0.001, 0.0001],
        'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'mlp_dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }

    accs = {}
    losses = {}

    kf = KFold(n_splits=4)

    for i, (train_index, test_index) in enumerate(kf.split(x)):
        x_train = x[train_index]
        y_train = y[train_index]
        x_val = x[test_index]
        y_val = y[test_index]

        for lr in hyperparameters['learning_rate']:
            for dr in hyperparameters['dropout']:
                for mlp_dr in hyperparameters['mlp_dropout']:
                    # open file for logs
                    f = open(f"./results/results_dr_{dr}_lr_{lr}_mlp_dr_{mlp_dr}.txt", "a+")

                    model = create_model(input_shape, mlp_dropout=mlp_dr, dropout=dr, learning_rate=lr)

                    callbacks = [keras.callbacks.EarlyStopping(patience=200, restore_best_weights=True)]

                    results = model.fit(
                        x_train,
                        y_train,
                        validation_data=(x_val,y_val),
                        epochs=10000,
                        batch_size=256,
                        callbacks=callbacks,
                        verbose=0
                    )

                    print(f"dr_{dr}_lr_{lr}_mlp_dr_{mlp_dr}")

                    f.write(str(results.history["val_sparse_categorical_accuracy"]))
                    f.write(str(results.history["val_loss"]))

                    l, a = model.evaluate(x_val, y_val, verbose=1)

                    if i == 0:
                        accs[(lr, dr, mlp_dr)] = a
                        losses[(lr, dr, mlp_dr)] = l
                    
                    else:
                        accs[(lr, dr, mlp_dr)] += a
                        losses[(lr, dr, mlp_dr)] += l
                    
                    if i==3:
                        accs[(lr, dr, mlp_dr)] /= 4
                        losses[(lr, dr, mlp_dr)] /= 4

                    f.close()


    best_hyperparameters = min(losses, key=losses.get)
    best_loss = losses[best_hyperparameters]
    best_accuracy = accs[best_hyperparameters]

    f = open("./results/final_results.txt", "w")
    print("Best Hyperparameters: (lr, dr, ks, ncv) -> ", best_hyperparameters)
    f.write(f"Best Hyperparameters: (lr, dr, ks, ncv) -> {best_hyperparameters}\n")
    print("Best Loss: ", best_loss)
    f.write(f"Best Loss: {best_loss}\n")
    print("Best Accuracy: ", best_accuracy)
    f.write(f"Best Accuracy: {best_accuracy}\n")
    f.close()
