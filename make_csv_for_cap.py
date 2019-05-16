import numpy as np

train_data=np.load('embedded_train_data.npy')
train_labels=np.load('train_labels.npy')
test_labels=np.load('test_labels.npy')
test_data=np.load('embedded_test_data.npy')
print(train_data.shape,train_labels.shape)
train_csv=np.concatenate((train_data,train_labels.reshape(train_labels.shape[0],1)),axis=1)
test_csv=np.concatenate((test_data,test_labels.reshape(test_labels.shape[0],1)),axis=1)
np.savetxt("train.csv", train_csv, delimiter=",")
np.savetxt("test.csv", test_csv, delimiter=",")