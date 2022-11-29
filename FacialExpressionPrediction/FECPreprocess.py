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
    transforms.RandomCrop((224,224), padding=4, padding_mode='reflect'),
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.01, 0.12),
        shear=(0.01, 0.03),
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5075, 0.5075, 0.5075],[0.2503, 0.2503, 0.2503])
])

transforms_validation = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5075, 0.5075, 0.5075],[0.2503, 0.2503, 0.2503])
])

data_path = 'Data/FER2013'
train_dataset = datasets.ImageFolder(os.path.join(data_path,'train'), transforms_train)
validation_dataset = datasets.ImageFolder(os.path.join(data_path,'test'), transforms_validation)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=16, shuffle=True)

print('Train dataset size:', len(train_dataset))
print('Validation dataset size:', len(validation_dataset))
class_names = train_dataset.classes
print('Class names:', class_names)
