import os
import torch
import torch.nn as nn
from numpy import float32
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from tqdm import tqdm

from FECPreprocess import train_dataset, validation_dataset, train_dataloader, validation_dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object

def validateModel():
    model_path = 'FEC1.pth'

    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 7)
    model.load_state_dict(torch.load(os.path.join(model_path), map_location=torch.device(device)))
    model.to(device)

    model.eval()


    y_pred = []
    y_true = []

    for data in tqdm(validation_dataloader):
        images, labels = data[0].to(device), data[1]
        y_true.extend(labels.numpy())

        outputs = model(images)


        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())

    np.save('gt_pred', np.array([y_true, y_pred]))

#validateModel()

predictions = np.load(os.path.join('gt_pred.npy'))
y_true, y_pred = np.array(predictions[0], dtype=float).tolist(), np.array(predictions[1], dtype=float).tolist()

cf_matrix = confusion_matrix(y_true,y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index= [i for i in train_dataset.classes], columns= [i for i in train_dataset.classes])
#df_cm = pd.DataFrame(cf_matrix, index= [i for i in train_dataset.classes], columns= [i for i in train_dataset.classes])

plt.figure(figsize= (12,7))
sn.heatmap(df_cm, annot=True)

# Plot the confusion matrix.
plt.ylabel('Actual', fontsize=13)
plt.xlabel('Prediction', fontsize=13)
plt.title('Confusion Matrix', fontsize=17)
plt.show()

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy   :", accuracy)
print(classification_report(y_true, y_pred, target_names=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']))
print(classification_report(y_true, y_pred, labels=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], encoded_labels=True, as_frame=True))

plt.savefig('output_cf_matrix.png')