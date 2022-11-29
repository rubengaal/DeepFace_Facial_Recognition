import torch
import torchvision
from torchvision import transforms
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object

transforms_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.6692, 0.4936, 0.4145],[0.2237, 0.1996, 0.1920])
])

transforms_validation = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.6692, 0.4936, 0.4145],[0.2237, 0.1996, 0.1920])
])

data_path = 'data/'
train_dataset = datasets.ImageFolder(os.path.join(data_path,'Training'), transforms_train)
validation_dataset = datasets.ImageFolder(os.path.join(data_path,'Validation'), transforms_validation)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=16, shuffle=True)

print('Train dataset size:', len(train_dataset))
print('Validation dataset size:', len(validation_dataset))
class_names = train_dataset.classes
print('Class names:', class_names)
