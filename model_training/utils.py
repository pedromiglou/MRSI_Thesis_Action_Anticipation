import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

from collections import Counter
from sklearn.metrics import confusion_matrix


# Set the font type to be used for plotting
plt.rcParams['svg.fonttype'] = 'none'


def read_data(folder_path, people=["joel", "manuel", "pedro"]):
    x = []
    y = []

    objects = {"bottle":0, "cube":1, "phone":2, "screwdriver":3}

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Create the absolute path to the file
        file_path = os.path.join(folder_path, filename)

        # Check if the file path is a file (not a directory)
        if os.path.isfile(file_path) and filename.split("_")[1] in people:
            
            f = open(file_path, "r")

            for line in f.readlines():
                ps = [[aux for aux in p[1:-2].split(" ") if len(aux)>0] for p in line.split(",")]
                x.append([[float(p[0]), float(p[1]), float(p[2])] for p in ps])
                y.append(objects[filename.split("_")[0]])

            f.close()
    
    x = np.array(x)
    #x = x[:,:,0:2]
    y = np.array(y)

    return x, y


def write_results(train_acc, val_acc, test_acc, train_loss, val_loss, test_loss,
        report, save_path="./results/results.txt"):
    
    f = open(save_path, "w")
    f.write(f"Training loss: {train_loss}\nTraining accuracy: {train_acc}\n")
    f.write(f"Validation loss: {val_loss}\nValidation accuracy: {val_acc}\n")
    f.write(f"Test loss: {test_loss}\nTest accuracy: {test_acc}\n")
    f.write(report)
    f.close()


#################################### Plots ####################################

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
