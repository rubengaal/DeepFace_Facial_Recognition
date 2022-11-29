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
from FECPreprocess import class_names

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
    transforms.Resize((224,224)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5075, 0.5075, 0.5075],[0.2503, 0.2503, 0.2503])
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
    model.fc = nn.Linear(num_features, 7)
    model.load_state_dict(torch.load('FEC1.pth'))
    model.to(device)

    model.eval()

    with torch.no_grad():

        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted]

print(PredictGender('Test/rubenneutral.jpg')) #NEUTRAL
print(PredictGender('Test/rubenangry.jpg')) #ANGRY
print(PredictGender('Test/huadisgust.jpg')) #DISGUST
print(PredictGender('Test/huafear.jpg'))    #FEAR
print(PredictGender('Test/rubensad.jpg'))   #SAD
print(PredictGender('Test/rubenhappy.jpg')) #HAPPY
print(PredictGender('Test/huasurprise.jpg'))#SURPRISE




