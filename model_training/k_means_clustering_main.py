import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


from utils import *


N_CLASSES = 4

if __name__ == "__main__":
    # read data
    x, y = read_dataset2()

    x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

    # data shuffling
    x, y = shuffle(x, y, random_state=0)

    # data splitting
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=1/5, random_state=0, stratify=y)

    # Create and fit the KMeans model
    kmeans = KMeans(n_clusters=4, n_init=1000)
    kmeans.fit(x_test)

    # Get the cluster assignments for each data point
    labels = kmeans.labels_

    labels = kmeans.predict(x_test)
    
    plot_confusion_matrix(confusion_matrix(y_test, labels), target_names=["bottle", "cube", "phone", "screw."],
                          show=True, save_path="./kmeans_conf_matrix.svg")
