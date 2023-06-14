import itertools
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def split_and_shuffle(x, y, balanced=False):
    # shuffle
    num = len(y)

    idx = np.random.permutation(num)

    x = x[idx]
    y = y[idx]

    # split
    num_labels = len(np.unique(y))

    while True:
        x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=1/5)

        if not balanced:
            break
        
        label_counts = Counter(y_test)

        counts = list(label_counts.values())

        if max(counts) - min(counts)<2 and len(counts)==num_labels:
            break
    
    while True:
        x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=1/4)

        if not balanced:
            break

        label_counts = Counter(y_val)

        counts = list(label_counts.values())

        if max(counts) - min(counts)<2 and len(counts)==num_labels:
            break
    
    return x_train, y_train, x_val, y_val, x_test, y_test


#################################### Plots ####################################

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


def plot_confusion_matrix(y_test, y_pred, target_names=None):
    cm = confusion_matrix(y_test, y_pred)

    fig = plt.figure(figsize=(10, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))

    plt.colorbar()

    ax = fig.gca()

    ax.tick_params(bottom=False, top=True, left=True, right=False)
    ax.tick_params(labelbottom=False, labeltop=True, labelleft=True, labelright=False)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label', fontsize=14)
    plt.title('Predicted Label', fontsize=14)
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


def draw_bar_chart(labels):
    fig, ax = plt.subplots(figsize=(10, 5))
    # Count the frequency of each label
    label_counts = Counter(labels)

    # Get the labels and their respective counts
    labels = list(label_counts.keys())
    counts = list(label_counts.values())

    # Set up the bar chart
    plt.bar(labels, counts)
    ax.set_xticks(labels)
    plt.xlabel('Labels')
    plt.ylabel('Quantity')
    plt.title('Quantity of Labels from Each Class')

    # Display the chart
    plt.show()
