import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter


def split_and_shuffle2(X, y):
    label_counts = Counter(y)
    num_labels = len(label_counts.values())

    while True:
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=1/10)
        label_counts = Counter(y_test)

        counts = list(label_counts.values())

        if max(counts) - min(counts)<2 and len(counts)==num_labels:
            break
    
    while True:
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/9)
        label_counts = Counter(y_val)

        counts = list(label_counts.values())

        if max(counts) - min(counts)<2 and len(counts)==num_labels:
            break
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def split_and_shuffle(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/10)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/9)

    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_accuracy_comparison(accs, title, legend):
    epochs = len(accs[0])
    plt.figure(figsize = (10,5))
    for acc in accs:
        plt.plot(range(1, epochs+1), acc)

    plt.xticks(range(1, epochs+1))
    plt.title(title)
    plt.legend(legend)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()


def plot_loss_comparison(losses, title, legend):
    epochs = len(losses[0])
    plt.figure(figsize = (10,5))
    for loss in losses:
        plt.plot(range(1, epochs+1), loss)

    plt.xticks(range(1, epochs+1))
    plt.title(title)
    plt.legend(legend)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    plt.matshow(confusion_matrix(y_test, y_pred))
    plt.ylabel("Predicted Category", fontsize=14)
    plt.title("Category", fontsize=14)
    plt.show()