import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import *

folder_path = './points'

x, y = read_data(folder_path)

n_classes = len(np.unique(y))

x_train, y_train, x_val, y_val, x_test, y_test = split_and_shuffle(x, y, balanced=False)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

input_shape = x.shape[1:]

batch_sizes = [16, 32, 64, 128, 256]
acc = []
loss = []
times = []

for b_size in batch_sizes:
    model = build_model(
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["sparse_categorical_accuracy"]
    )

    callbacks = [keras.callbacks.EarlyStopping(patience=250, restore_best_weights=True)]

    results = model.fit(
        x_train,
        y_train,
        validation_data=(x_val,y_val),
        epochs=2000,
        batch_size=b_size,
        callbacks=callbacks,
    )

    acc.append(results.history["val_sparse_categorical_accuracy"])
    loss.append(results.history["val_loss"])

plot_accuracy_comparison(acc, "Batch Size Comparison (Validation Accuracy)", batch_sizes,
                        show=False, save_path = "./results/acc_comparison.png")

plot_loss_comparison(loss, "Batch Size Comparison (Validation Loss)", batch_sizes,
                    show=False, save_path = "./results/loss_comparison.png")
