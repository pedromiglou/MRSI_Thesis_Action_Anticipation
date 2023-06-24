import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils import *

folder_path = './points'

x, y = read_data2(folder_path)

n_classes = len(np.unique(y))

x_train, y_train, x_val, y_val, x_test, y_test = split_and_shuffle(x, y, balanced=False)

def fc_model(dropout=0.2):
   # Create a `Sequential` model and add a Dense layer as the first layer.
   model = tf.keras.models.Sequential()
   model.add(tf.keras.Input(shape=(5,5,3)))
   model.add(tf.keras.layers.Conv2D(32, 2, activation='relu'))
   model.add(tf.keras.layers.MaxPooling2D(2,1))
   model.add(tf.keras.layers.Flatten())
   model.add(tf.keras.layers.Dropout(dropout))
   model.add(tf.keras.layers.Dense(16, activation='relu'))
   model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))
   return model

model = fc_model()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["sparse_categorical_accuracy"]
)

callbacks = [keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True),
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
    epochs=1000,
    batch_size=16,
    callbacks=callbacks,
)

model.evaluate(x_test, y_test, verbose=1)

plot_accuracy_comparison([results.history["sparse_categorical_accuracy"], results.history["val_sparse_categorical_accuracy"]],
                        "Training/Validation Accuracy Comparison",
                        ["Training Accuracy", "Validation Accuracy"],
                        show=False, save_path = "./results/acc_comparison.png")

plot_loss_comparison([results.history["loss"], results.history["val_loss"]],
                     "Training/Validation Loss Comparison",
                     ["Training Loss", "Validation Loss"],
                     show=False, save_path = "./results/loss_comparison.png")

y_pred=np.argmax(model.predict(x_test), axis=-1)
plot_confusion_matrix(y_test, y_pred, ["bottle", "cube", "phone", "screwdriver"],
                      show=False, save_path = "./results/conf_matrix.png")