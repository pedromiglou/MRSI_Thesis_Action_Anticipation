import numpy as np
import os
import random
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras
from utils import *

from cnn_main import create_model


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
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=1/5, random_state=0, stratify=y)

    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=1/4, random_state=0, stratify=y_temp, shuffle=True)

    batch_sizes = [16, 32, 64, 128]
    acc = []
    loss = []

    for b_size in batch_sizes:
        print(f"testing {b_size} batch size")

        model = create_model(input_shape)

        callbacks = [keras.callbacks.EarlyStopping(patience=200, restore_best_weights=True)]

        results = model.fit(
            x_train,
            y_train,
            validation_data=(x_val,y_val),
            epochs=10000,
            batch_size=b_size,
            callbacks=callbacks,
            verbose=0
        )

        acc.append(results.history["val_sparse_categorical_accuracy"])
        loss.append(results.history["val_loss"])

    plot_accuracy_comparison(acc, "Batch Size Comparison (Validation Accuracy)", batch_sizes,
                            show=False, save_path = "./results/acc_comparison.png")

    plot_loss_comparison(loss, "Batch Size Comparison (Validation Loss)", batch_sizes,
                        show=False, save_path = "./results/loss_comparison.png")
