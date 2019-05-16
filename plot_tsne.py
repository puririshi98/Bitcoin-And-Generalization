import numpy as np
import pylab

Y = np.load("tsne_10000.npy")
labels = np.load("train_labels.npy")
pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)