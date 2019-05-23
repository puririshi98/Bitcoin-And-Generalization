from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


# model = nn.Sequential(nn.Linear(512, 40),
# 					nn.ReLU(),
					 
# 					nn.Linear(40,2)

					
# 					)

model = nn.Sequential(nn.Linear(512, 2),
					

					
					)
# model=torch.load('bestres_last_layer')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']
# optimizer_ft = optim.SGD(model.parameters(), lr=0.001)
optimizer_ft = optim.SGD(model.parameters(), lr=0.001)
train_data=np.load('embedded_train_data.npy')
train_labels=np.load('train_labels.npy')

indy=np.random.randint(0, train_labels.shape[0], size=int(train_labels.shape[0]/4))
train_labels=train_labels[indy]
train_data=train_data[indy]
test_labels=np.load('test_labels.npy')

test_data=np.load('embedded_test_data.npy')
batch_size=int(32)
datalens={'train':int(train_data.shape[0]/batch_size),'test':int(test_data.shape[0]/batch_size)}
# Decay LR by a factor of 0.1 every 7 epochs
plateau_scheduler=lr_scheduler.ReduceLROnPlateau(optimizer_ft,mode='max',factor=.9,patience=4)
def make_batch(batch_size,dataset,labels,phase):
	indy=np.random.randint(0,datalens[phase]*batch_size,size=batch_size)
	batch=dataset[indy]
	label=labels[indy]
	return batch,label
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.7)
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
	since = time.time()
   
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	best_train_acc=0.0
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'test']:
			if phase == 'train':
				scheduler.step(best_train_acc)
				print(get_lr(optimizer_ft))
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data.
			for i in range(10000):
				# if i%10000==0:
				# 	print(i)
				if phase=='train':
					inputs,label=make_batch(batch_size,train_data,train_labels,phase)
					#inputs=torch.tensor(train_data[indy],dtype=torch.float)
					#label=train_labels[indy]
					inputs=torch.tensor(inputs,dtype=torch.float)
					labels=torch.tensor(label,dtype=torch.long)
				else:
					# inputs=torch.tensor(test_data[i],dtype=torch.float)
					# labels=torch.tensor(test_labels[i],dtype=torch.long)
					inputs,label=make_batch(batch_size,test_data,test_labels,phase)
					inputs=torch.tensor(inputs,dtype=torch.float)
					labels=torch.tensor(label,dtype=torch.long)
				inputs = inputs.to(device)
				labels = labels.reshape(batch_size).to(device)
		
				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs).reshape(batch_size,2)
					#print(outputs)
					#print(labels.shape,outputs.shape)
					_, preds = torch.max(outputs, 1)
					#print(preds)
					loss = criterion(outputs, labels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()


				# statistics
				running_loss += loss.item() * inputs.size(0)
				# if preds==labels:
				# 	#print('hi')

				# 	running_corrects += 1
				running_corrects+=torch.sum(preds==labels.data)
			# epoch_loss = running_loss / (datalens[phase])
			
			# epoch_acc = running_corrects.float() / (datalens[phase]*batch_size)
			epoch_loss = running_loss / 320000
		
			epoch_acc = running_corrects.float() / 320000
			#print(running_corrects,datalens[phase])
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))
			if phase == 'train' and epoch_acc > best_train_acc:
				best_train_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())
			# deep copy the model
			if phase == 'test' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model
model = train_model(model, criterion, optimizer_ft, plateau_scheduler,num_epochs=1000)
torch.save(model,'bestfullyconnected_512_2_ensemble1')