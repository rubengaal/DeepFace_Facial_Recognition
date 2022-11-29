import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import pandas as pd
import seaborn as sn
import os

from tqdm import tqdm

from FECPreprocess import train_dataset, validation_dataset, train_dataloader, validation_dataloader




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object
model = models.resnet18(pretrained=True)
feature_no = model.fc.in_features
model.fc = nn.Linear(feature_no, 7)
model = model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epochs = 7
start_time = time.time()


for epoch in range(epochs):
    #TRAINING
    model.train()

    running_loss = 0
    running_corrects = 0

    print('Epoch-{0} lr: {1}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs,1)
        loss = loss_function(outputs,labels)


        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss/ len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset) * 100
    print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))

    #VALIDATION

    model.eval()

    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0

        for inputs, labels in validation_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)


            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(validation_dataset)
        epoch_acc = running_corrects / len(validation_dataset) * 100
        print('[Validation #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc,time.time() - start_time))

save_path = 'FEC1.pth'
torch.save(model.state_dict(),save_path)


y_pred = []
y_true = []

for data in tqdm(validation_dataloader):
    images, labels = data[0].to(device), data[1]
    y_true.extend(labels.numpy())

    outputs = model(images)


    _, predicted = torch.max(outputs, 1)
    y_pred.extend(predicted.cpu().numpy())

cf_matrix = confusion_matrix(y_true,y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index= [i for i in train_dataset.classes], columns= [i for i in train_dataset.classes])
plt.figure(figsize= (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output_cf_matrix.png')









