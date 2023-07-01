import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils import *
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

folder_path = './points'

x, y = read_data(folder_path)

n_classes = len(np.unique(y))

x, x_test, y, y_test = train_test_split(x, y, test_size=1/5)

def create_model(dropout=0.5, learning_rate=0.0001, kernel_size=2, num_conv_layers=3):
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

hyperparameters = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'kernel_size': [2, 3],
    'num_conv_layers': [1,2,3]
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
            for ks in hyperparameters['kernel_size']:
                for ncv in hyperparameters['num_conv_layers']:
                    # open file for logs
                    f = open(f"./results/results_dr_{dr}_lr_{lr}_ks_{ks}_ncv_{ncv}.txt", "a+")

                    model = create_model(dropout=dr, learning_rate=lr, kernel_size=ks, num_conv_layers=ncv)

                    callbacks = [keras.callbacks.EarlyStopping(patience=200, restore_best_weights=True)]

                    results = model.fit(
                        x_train,
                        y_train,
                        validation_data=(x_val,y_val),
                        epochs=10000,
                        batch_size=128,
                        callbacks=callbacks,
                        verbose=0
                    )

                    print(f"dr_{dr}_lr_{lr}_ks_{ks}_ncv_{ncv}")

                    f.write(str(results.history["val_sparse_categorical_accuracy"]))
                    f.write(str(results.history["val_loss"]))

                    l, a = model.evaluate(x_val, y_val, verbose=1)

                    if i == 0:
                        accs[(lr, dr, ks, ncv)] = a
                        losses[(lr, dr, ks, ncv)] = l
                    
                    else:
                        accs[(lr, dr, ks, ncv)] += a
                        losses[(lr, dr, ks, ncv)] += l
                    
                    if i==3:
                        accs[(lr, dr, ks, ncv)] /= 4
                        losses[(lr, dr, ks, ncv)] /= 4

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


"""
# Define the callbacks
callbacks = [keras.callbacks.EarlyStopping(patience=200, restore_best_weights=True)]

# Create the TensorFlow model
model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, epochs=1, batch_size=128, verbose=0, callbacks=callbacks)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=hyperparameters, cv=4, scoring="neg_log_loss")
grid_search.fit(x, y, validation_split=0.2)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)
"""