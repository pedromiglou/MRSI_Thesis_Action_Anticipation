import itertools
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from sklearn.metrics import confusion_matrix


# Set the font type to be used for plotting
plt.rcParams['svg.fonttype'] = 'none'


def plot_accuracy_comparison(accs, title, legend, show=True, save_path=False):
    plt.figure(figsize = (10,5))
    for acc in accs:
        plt.plot(range(1, len(acc)+1), acc)

    #plt.xticks(range(1, epochs+1))
    plt.title(title)
    plt.legend(legend)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    
    if show:
        plt.show()


def plot_confusion_matrix(y_test, y_pred, target_names=None, show=True, save_path=False):
    cm = confusion_matrix(y_test, y_pred)

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=13)

    ax = fig.gca()

    ax.tick_params(bottom=False, top=True, left=True, right=False)
    ax.tick_params(labelbottom=False, labeltop=True, labelleft=True, labelright=False)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, fontsize=13)
        plt.yticks(tick_marks, target_names, fontsize=13, rotation="vertical", va="center")

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=13)

    plt.ylabel('True Label', fontsize=16)
    plt.title('Predicted Label', fontsize=16)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path) # , format="svg")
    
    if show:
        plt.show()


def plot_loss_comparison(losses, title, legend, show=True, save_path=False):
    plt.figure(figsize = (10,5))
    for loss in losses:
        plt.plot(range(1, len(loss)+1), loss)

    #plt.xticks(range(1, epochs+1))
    plt.title(title)
    plt.legend(legend)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    
    if show:
        plt.show()


def draw_bar_chart(labels, show=True, save_path=False):
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
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    
    if show:
        plt.show()
