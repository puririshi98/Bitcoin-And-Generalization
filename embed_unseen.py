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
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (10000, rlimit[1]))
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(3),
        transforms.RandomRotation(3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'final_unseen_test/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.nn.Module.dump_patches = True
model=torch.load('bestres',map_location='cpu')
model.to(device)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'test']}
features = model._modules.get('avgpool')
model.eval()
def get_vector(input):
    
    my_embedding = torch.zeros(1,512,1,1)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    # 5. Attach that function to our selected layer
    h = features.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    input=input.to(device)
    model(input)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding.reshape(512)

train_data=[]
train_labels=[]
test_data=[]
test_labels=[]
i=0
# for inputs,labels in dataloaders['train']:
#     i+=1
#     if i%1000==0:
#         print(i)  
#     feat=get_vector(inputs).numpy().tolist()
#     train_data.append(feat)
#     train_labels.append(labels)
for inputs,labels in dataloaders['test']:
    feat=get_vector(inputs).numpy().tolist()
    test_data.append(feat)
    test_labels.append(labels)
train_data=np.array(train_data)
train_labels=np.array(train_labels)
test_data=np.array(test_data)
test_labels=np.array(test_labels)
np.save('embedded_unseen_test_data.npy',test_data)
np.save('unseen_test_labels.npy',test_labels)
# np.save('embedded_train_data.npy',train_data)
# np.save('train_labels.npy',train_labels)
# np.save('embedded_test_data.npy',test_data)
# np.save('test_labels.npy',test_labels)
print('done')
