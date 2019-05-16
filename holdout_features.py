from capacityreq import cap
import numpy as np
train_data=np.load('embedded_train_data.npy')
train_labels=np.load('train_labels.npy')
batch_size=int(10000)
indy=np.random.randint(0, train_labels.shape[0], size=batch_size)
train_labels=train_labels[indy]
train_data=train_data[indy]
bestmemeqcap=9999999999
best_feat_nums=None
num_feats=10
for i in range(1000):
	
	
	
	feat_nums=np.random.randint(0, train_data.shape[1], size=num_feats)
	featurized=train_data[:,feat_nums]
	print(featurized.shape, train_labels.shape)
	train_csv=np.concatenate((featurized,train_labels.reshape(train_labels.shape[0],1)),axis=1)
	print(feat_nums)
	memeqcap=cap(train_csv)
	if memeqcap<bestmemeqcap:
		bestmemeqcap=memeqcap
		best_feat_nums=feat_nums
print(best_feat_nums)
print(bestmemeqcap)