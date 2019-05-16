import torch 
import numpy as np

test_labels=np.load('unseen_test_labels.npy')

test_data=np.load('embedded_unseen_test_data.npy')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def make_batch(batch_size,dataset,labels):
		indy=np.random.randint(0,test_labels.shape[0],size=batch_size)
		batch=dataset[indy]
		label=labels[indy]
		return batch,label
model_preds=[[] for i in range(test_data.shape[0])]


model_list=[torch.load('bestfullyconnected_512_18_ensemble'+str(num_model)) for num_model in range(1,16)]
correct=0.0
for i in range(10000):
	data,target=make_batch(32,test_data,test_labels)
		# if cuda:
		#     data, target = data.cuda(), target.cuda()
	data, target = torch.tensor(data,dtype=torch.float), torch.tensor(target,dtype=torch.int)
	output_list = [model(data).reshape(32,2) for model in model_list]

	pred_list=[output.data.max(1)[1].reshape(1,-1) for output in output_list]
	#output=vote(output_list)
	# print(target)
	# print(pred_list)
	#print(pred_list)
	output = torch.sum(torch.cat(pred_list), 0)
	#print(output)
	output=torch.tensor(output>=8,dtype=torch.int)
	#print(output)
	#print(target)
		#test_loss += F.nll_loss(output, target).data[0]
	
	# print(output)
	# print(target)
	correct += torch.sum(output==target)
print(correct)
print(10000*32)
# for i in range(100):
# 	inputty,labely=torch.tensor(make_batch(32,test_data,test_labels),dtype=torch.float)
# 	for num_model in range(1,10):
# 		model=torch.load('bestfullyconnected_512_40_ensemble'+str(num_model))
		
# 		pred=model(inputty).reshape(32,2)
# 		_, pred = torch.max(pred, 1)
		
# 		model_preds[i].append(pred.item())

# 	print('evald model'+str(num_model)+' accuracy=  '+str((float(correct)/test_data.shape[0])))
#eval
# count1=0
# num_correct=0
# for i,votes in enumerate(model_preds):
# 	labely=test_labels[i]
# 	for vote in votes:
# 		if vote==1:
# 			count1+=1

# 	if count1>=5:
# 		pred=1
# 	else:
# 		pred=0
# 	if pred==labely:
# 		num_correct+=1
# print("accuracy"+str(float(num_correct)/test_data.shape[0]))






		


