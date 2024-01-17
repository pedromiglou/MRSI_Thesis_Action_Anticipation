import numpy as np
import time

from functools import partial
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras

from transformer_main import create_model
from utils import *


def test_transformer(train_users, train_sessions, test_users=None, test_sessions=None,
        num_test_samples=None, n_simulations=1, test_name="transformer", test_params=None):
    sum_accs = 0
    sum_losses = 0
    sum_times = 0
    sum_precisions = 0
    sum_recalls = 0
    sum_f1_scores = 0
    sum_conf_matrices = np.zeros((4,4))

    for _ in range(n_simulations):
        # read data
        x_train, y_train = read_dataset2(sessions=train_sessions, people=train_users)

        input_shape = x_train.shape[1:]

        # data shuffling
        x_train, y_train = shuffle(x_train, y_train)

        # data splitting
        if test_users is None and test_sessions is None:
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=1/5, stratify=y_train, shuffle=True)
        elif test_users is not None and test_sessions is not None:
            x_test, y_test = read_dataset2(sessions=test_sessions, people=test_users)
        else:
            raise ValueError("test_users and test_sessions must be both None or not None")
        
        if num_test_samples is not None:
            random_indices = np.random.choice(len(y_test), size=num_test_samples, replace=False)

            x_test = x_test[random_indices]
            y_test = y_test[random_indices]

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=1/5, stratify=y_train, shuffle=True)

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

        cm = confusion_matrix(y_test, y_pred)
        sum_conf_matrices += cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    file_path = f"./results/{test_name}"
    if test_params is not None:
        if "user" in test_params:
            file_path += f"_{test_users[0] if test_users is not None else train_users[0]}"
        if "session" in test_params:
            file_path += f"_session_{test_sessions[0]}"


    f = open(file_path+"_results.txt", "w")
    f.write(f"Average accuracy: {sum_accs/n_simulations}")
    f.write(f"Average loss: {sum_losses/n_simulations}")
    f.write(f"Average time: {sum_times/n_simulations}")
    f.write(f"Average precision: {sum_precisions/n_simulations}")
    f.write(f"Average recall: {sum_recalls/n_simulations}")
    f.write(f"Average f1-score: {sum_f1_scores/n_simulations}")
    f.close()

    plot_confusion_matrix(sum_conf_matrices/n_simulations, ["bottle", "cube", "phone", "screw."],
                            show=False, save_path = file_path+"_conf_matrix.svg")


test_transformer_multi_user = partial(test_transformer, train_users = ["joel", "manuel", "pedro"], train_sessions=["1","2","3","4"],
                                      n_simulations=25, test_name="transformer_multi_user")

test_transformer_multi_user_by_session = partial(test_transformer, train_users = ["joel", "manuel", "pedro"], test_users=["joel", "manuel", "pedro"],
                                                 num_test_samples=2713, n_simulations=25, test_name="transformer_multi_user_by_session", test_params=["session"])

test_transformer_intra_user = partial(test_transformer, train_sessions=["1","2","3","4"], n_simulations=25,
                                      num_test_samples=732, test_name = "transformer_intra_user", test_params=["user"])

test_transformer_intra_user_by_session = partial(test_transformer, n_simulations=25, test_name = "transformer_intra_user_by_session",
                                                 num_test_samples=872, test_params=["session","user"])

test_transformer_inter_user = partial(test_transformer, train_sessions=["1","2","3","4"], test_sessions=["1","2","3","4"], n_simulations=25,
                                      num_test_samples=3663, test_name = "transformer_inter_user", test_params=["user"])

# tests
#test_transformer_multi_user_by_session(train_sessions=["2","3","4"], test_sessions=["1"])
#test_transformer_multi_user_by_session(train_sessions=["1","3","4"], test_sessions=["2"])
#test_transformer_multi_user_by_session(train_sessions=["1","2","4"], test_sessions=["3"])
#test_transformer_multi_user_by_session(train_sessions=["1","2","3"], test_sessions=["4"])

#for test_people, train_people in zip([["joel"], ["manuel"], ["pedro"]], [["manuel", "pedro"], ["joel", "pedro"], ["joel", "manuel"]]):
#    test_transformer_inter_user(train_users=train_people, test_users=test_people)

#for train_sessions, test_sessions in zip([["2","3","4"], ["1","3","4"]], [["1"],["2"]]):
#    test_transformer_intra_user_by_session(train_users=["joel"], test_users=["joel"], train_sessions=train_sessions, test_sessions=test_sessions)
#for train_sessions, test_sessions in zip([["1","2","4"], ["1","2","3"]], [["3"],["4"]]):
#    test_transformer_intra_user_by_session(train_users=["joel"], test_users=["joel"], train_sessions=train_sessions, test_sessions=test_sessions)

#test_transformer_intra_user(train_users=["joel"])
#test_transformer_intra_user(train_users=["manuel"])
#test_transformer_intra_user(train_users=["pedro"])

test_transformer_multi_user(n_simulations=12)
