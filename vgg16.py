#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#!pip install torch torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython import display
import zipfile as zf

#files = zf.ZipFile('dataset.zip', 'r')
#files.extractall(path=None)


# In[4]:


from os import listdir
trainingPath = "dataset/training_set"
validationPath = "dataset/validation_set"
labels = set(listdir(trainingPath))
print("Training path:", trainingPath)
print("Validation path:", validationPath)
print()
print("Labels:", labels)


# In[5]:


import os
# training_sculpt = os.listdir(trainingPath+'/'+'sculpture')
# training_draw = os.listdir(trainingPath+'/'+'drawings')
# training_icon = os.listdir(trainingPath+'/'+'iconography')
# training_paint = os.listdir(trainingPath+'/'+'painting')
# training_engrave = os.listdir(trainingPath+'/'+'engraving')

# val_sculpt = os.listdir(validationPath+'/'+'sculpture')
# val_draw = os.listdir(validationPath+'/'+'drawings')
# val_icon = os.listdir(validationPath+'/'+'iconography')
# val_paint = os.listdir(validationPath+'/'+'painting')
# val_engrave = os.listdir(validationPath+'/'+'engraving')
# for file in training_sculpt[100:]:
#     os.remove(file)
# for file in training_draw[100:]:
#     os.remove(file)
# for file in training_icon[100:]:
#     os.remove(file)
# for file in training_paint[100:]:
#     os.remove(file)
# for file in training_engrave[100:]:
#     os.remove(file)
# for file in val_sculpt[50:]:
#     os.remove(file)
# for file in val_draw[50:]:
#     os.remove(file)
# for file in val_icon[50:]:
#     os.remove(file)
# for file in val_paint[50:]:
#     os.remove(file)
# for file in val_engrave[50:]:
#     os.remove(file)


# In[6]:


from torchvision import transforms
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device\n")


# In[7]:


modelPath = "model.pth"

trainingPath = "dataset/training_set_2"
testingPath = "dataset/validation_set_2"

print("Training path:", trainingPath)
print("Testing  path:", testingPath)


# In[8]:


imgTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# In[9]:


trainingSet = torchvision.datasets.ImageFolder(trainingPath, transform=imgTransform)
testingSet = torchvision.datasets.ImageFolder(testingPath, transform=imgTransform)

batchSize = 10
trainLoader = torch.utils.data.DataLoader(trainingSet, batchSize)#, num_workers=2)
testLoader = torch.utils.data.DataLoader(testingSet, batchSize)# num_workers=2)

print("Sample Count:")
print(f"training {len(trainingSet)}")
print(f"testing  {len(testingSet)}")


# In[3]:


#pip install ipywidgets


# In[10]:


model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)


# In[11]:


get_ipython().system('pip install torchinfo')
import torchvision.models as models
from torchinfo import summary


# In[12]:


from torchvision import datasets


# In[13]:


vgg_16 = models.vgg16(pretrained=True)
vgg_16


# In[14]:


num_features = vgg_16.classifier[6].in_features
features = list(vgg_16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 5)]) # Add our layer with 5 outputs
vgg_16.classifier = nn.Sequential(*features) # Replace the model classifier
print(vgg_16)


# In[13]:


# import torch.optim as optim
# import torch.nn.functional as func
# for images, labels in trainLoader:
#     print(labels)
#     out = vgg_16(images)
# #     print(out)
#     probabilities = func.softmax(out[0], dim=0)
#     print(probabilities)
#     out = vgg_16(images)
#     print('out.shape:', out.shape)
#     print('out[0]:', out[0])
# for images, labels in testLoader:
#     out = vgg_16(images)
#     print(out[0])


# In[15]:


import torch.optim as optim
import torch.nn.functional as func


# In[16]:


epochs = 20
test_acc_lst = []
loss_lst = []
optimizer = optim.SGD(vgg_16.parameters(), lr = 0.001, momentum = 0.9)
for ep in range(epochs):
    total_loss = []
    for images, labels in iter(trainLoader):
        optimizer.zero_grad()
        outputs = vgg_16(images)
        loss = func.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss.append(loss)
    loss_lst.append(sum(total_loss)/len(total_loss))
    test_error = 0
    count = 0
    for images, labels in iter(testLoader):
        count += 1
        outputs = vgg_16(images)
        print(labels)
        for i in range(batchSize):
            max_val_list = func.softmax(outputs[i], dim=0).tolist() # Applies softmax - calculates probabilities of output being in a given class
            max_val = max(max_val_list) # finds max probability
            output1 = max_val_list.index(max_val) # index of max probability is the predicted class label
            print(output1)
            if labels[i] != output1:
                test_error += 1
    test_acc = 1.0-float(test_error)/float(len(testingSet))
    print('%d: %f' % (ep, test_acc))
    test_acc_lst.append(test_acc)
print(test_acc_lst)
print(loss_lst)


# In[ ]:




