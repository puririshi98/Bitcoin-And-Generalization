from capacityreq import cap
import numpy as np
train_data=np.load('embedded_train_data.npy')
train_labels=np.load('train_labels.npy')

for portion in range(10,11):
	
	print(str(portion*10)+'%')
	batch_size=int(portion/10.0*train_labels.shape[0])
	indices=list(np.random.randint(0, train_labels.shape[0], size=batch_size))
	batch=np.array([train_data[i] for i in indices])
	label_batch=np.array([train_labels[i] for i in indices])
	train_csv=np.concatenate((batch,label_batch.reshape(label_batch.shape[0],1)),axis=1)
	cap(train_csv)
	#np.savetxt(str(portion)+'_10'+"train.csv", train_csv, delimiter=",")
	#cap(str(portion)+'_10'+"train.csv")
