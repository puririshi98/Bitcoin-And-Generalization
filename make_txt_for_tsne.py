import numpy as np
train_data=np.load('embedded_train_data.npy')
train_labels=np.load('train_labels.npy')

np.savetxt("train_embed_tsne_in.txt", train_data, delimiter=" ")
np.savetxt("train_label_tsne_in.txt", train_labels, delimiter=" ")