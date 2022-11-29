import torch
import torch.nn as nn
from natsort import natsort
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from Preprocess import class_names
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object

def imshow(input, title):
    # torch.Tensor => numpy
    input = input.numpy().transpose((1, 2, 0))
    # undo image normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # display images
    plt.imshow(input)
    plt.title(title)
    plt.show()


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.6692, 0.4936, 0.4145],[0.2237, 0.1996, 0.1920])
])


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = transform(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU

def PredictGender(path):

    #image = image_loader('Test/test.jpg')
    image = image_loader(path)

    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load('GC1.pth'))
    model.to(device)

    model.eval()

    with torch.no_grad():

        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted]

print("EREDMÉNY: "+PredictGender('Test/rubenneutral.jpg'))

