import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils import *

folder_path = './points'

x, y = read_data(folder_path)

n_classes = len(np.unique(y))

x_train, y_train, x_val, y_val, x_test, y_test = split_and_shuffle(x, y, balanced=False)

def fc_model(dropout=0.4, learning_rate=0.0001, kernel_size=2, num_conv_layers=2):
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


batch_sizes = [16, 32, 64, 128]
acc = []
loss = []
times = []

for b_size in batch_sizes:
    model = fc_model()

    callbacks = [keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)]

    results = model.fit(
        x_train,
        y_train,
        validation_data=(x_val,y_val),
        epochs=1000,
        batch_size=b_size,
        callbacks=callbacks,
    )

    acc.append(results.history["val_sparse_categorical_accuracy"])
    loss.append(results.history["val_loss"])

plot_accuracy_comparison(acc, "Batch Size Comparison (Validation Accuracy)", batch_sizes,
                        show=False, save_path = "./results/acc_comparison.png")

plot_loss_comparison(loss, "Batch Size Comparison (Validation Loss)", batch_sizes,
                    show=False, save_path = "./results/loss_comparison.png")
