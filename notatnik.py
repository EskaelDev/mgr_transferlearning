# -*- coding: utf-8 -*-
"""notatnik.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GbpCvRFK71JPhdruY4deNcdEnWxxI62d

Clone Repo
"""

!git clone https://github.com/Natsyu/mgr_transferlearning.git git-modules

"""Remove cloned repo"""

import shutil
def rmgit():
    shutil.rmtree('/content/git-modules')

"""Add cloned files to path"""

import sys
sys.path.append('/content/git-modules')

from google.colab import output

"""# Setup Data

Mount GDrive
"""

from google.colab import drive
drive.mount('/content/drive',  force_remount=True)

"""## Copy datasets to colab"""

!cp /content/drive/MyDrive/RESISC45.ZIP /content/

!mkdir dataset
!mkdir /content/dataset/RESISC45
!mkdir /content/dataset/RESISC45/Images

!unzip /content/RESISC45.ZIP -d /content/dataset
!mv /content/dataset/NWPU-RESISC45/* /content/dataset/RESISC45/Images/

# !rsync -a -r --progress /content/drive/MyDrive/dataset /content

# !cp -R /content/drive/MyDrive/dataset /content

"""Import datasets"""

from datasetmodel import uc_landuse_ds, resisc_ds

"""Select dataset"""

working_ds = resisc_ds
print(len(working_ds.classes))

# working_ds.path = "/content/dataset/RESISC45/Images"

# Commented out IPython magic to ensure Python compatibility.
import os
import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from termcolor import colored
from pathlib import Path
from datetime import datetime

# %matplotlib inline

"""unzip cifar state dict"""

from unzipcifar import unzip_cifar
if not os.path.isdir('/content/git-modules/cifar10_models/state_dicts'):
    unzip_cifar()

"""## CUDA check"""

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('❌CUDA is not available.  Training on CPU ...')
else:
    print('✅CUDA is available!  Training on GPU ...')

"""### Models"""

from netmodels import TrainedModels, get_model



# model_selection = TrainedModels.vgg13_bn
# model_selection = TrainedModels.densenet169
# model_selection = TrainedModels.resnet18
model_selection = TrainedModels.mobilenet_v2

# rmgit()

"""### fold"""

import torch.nn as nn

model, input_size, mean, std, optimizer = get_model(model_selection, working_ds.class_num, train_on_gpu)
criterion = nn.CrossEntropyLoss()
n_epochs = 1000
max_no_improve_epochs = 20

model.name

"""## Load and Transform Data"""

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 50
confusion_batch_size = 100
# percentage of training set to use as validation
train_portion = 0.7
validation_portion = 0.2
test_portion = 0.1
# image_size = 255
if round(train_portion + validation_portion + test_portion) != 1:
    print(colored('❌Wrong sizes', 'red'))
else:
    print(colored('✅Sizes match', 'green'))

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.Resize((input_size, input_size)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(360),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean, std)])

# choose the training and test datasets
dataset = datasets.ImageFolder(working_ds.path, transform=transform)



train_length = int(len(dataset.imgs)*train_portion)
# train_length -= train_length % batch_size

validation_length = int(len(dataset.imgs)*validation_portion)
# validation_length -= validation_length % batch_size

test_length = int(len(dataset.imgs)*test_portion)
# test_length -= test_length % batch_size

# check sets lengsths
print(f'train_length:       {train_length}')
print(f'validation_length:  {validation_length}')
print(f'test_length:        {test_length}')
print(f'dataset length:     {len(dataset.imgs)}')
# if test_length + train_length + validation_length == len(dataset.imgs):
#     print('dataset divisible by batch_size')
# else:
#     print('dataset not divisible by batch_size')


# split and load data
train_set, validation_set, test_set, _ = torch.utils.data.random_split(dataset, [train_length, validation_length, test_length, len(dataset) - (test_length + train_length + validation_length)])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
confusion_loader = torch.utils.data.DataLoader(test_set, batch_size=confusion_batch_size, num_workers=num_workers)

print('Train loader:    ', len(train_loader))
print('Valid loader:    ', len(validation_loader))
print('Test loader:     ', len(test_loader))

"""### DataLoaders and Data Visualization"""

# # Visualize some sample data

# # obtain one batch of training images
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
# images = images.numpy() # convert images to numpy for display

# # plot the images in the batch, along with the corresponding labels
# fig = plt.figure(figsize=(25, 4))
# for idx in np.arange(20):
#     ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
#     plt.imshow(np.transpose(images[idx], (1, 2, 0)))
#     ax.set_title(working_ds.classes[labels[idx]])

"""### Specify [Loss Function](http://pytorch.org/docs/stable/nn.html#loss-functions) and [Optimizer](http://pytorch.org/docs/stable/optim.html)

Below we'll use cross-entropy loss and stochastic gradient descent with a small learning rate. Note that the optimizer accepts as input _only_ the trainable parameters `vgg.classifier.parameters()`.
"""

# import torch.optim as optim
# import torch.nn as nn
# # specify loss function (categorical cross-entropy)
# criterion = nn.CrossEntropyLoss()

# # specify optimizer (stochastic gradient descent) and learning rate = 0.001
# optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

"""Set all network parameters"""

from netparams import NetParams
netparams = NetParams(model=model,
                      criterion=criterion,
                      optimizer=optimizer,
                      n_epochs=n_epochs,
                      confusion_loader=confusion_loader,
                      max_no_improve_epochs=max_no_improve_epochs,
                      train_on_gpu=train_on_gpu,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      validation_loader=validation_loader,
                      batch_size=batch_size)

# rmgit()

"""# Train"""

from NetworkHelpers import train_loop, plot_array, test_model, plot_test_results, confusion, loop_inplace_sum, get_plot_confusion, recall_precision_fmeasure
from trainstats import TrainStats

train_stats: TrainStats = train_loop(netparams)

netparams.model.load_state_dict(torch.load('model_cifar (1).pt'))

netparams.model.name

train_stats.best_epoch

"""### Save trainig params to file"""

# plot_array('Train loss', train_stats.train_loss_array, train_stats.best_epoch)

# plot_array('Train acc', train_stats.train_accuracy_array, train_stats.best_epoch)

# plot_array('Valid loss', train_stats.valid_loss_array, train_stats.best_epoch)

# plot_array('Valid acc', train_stats.valid_accuracy_array, train_stats.best_epoch)

"""# Test"""

train_stats.accuracy, train_stats.k1, train_stats.k5, train_stats.class_accuracy = test_model(netparams, working_ds)

# confusion_matrix = torch.zeros(working_ds.class_num, working_ds.class_num)
# with torch.no_grad():
#     netparams.model.cpu()
#     for i, (inputs, classes) in enumerate(confusion_loader):
#         inputs = inputs.cpu()
#         classes = classes.cpu()
#         outputs = netparams.model(inputs)
#         _, preds = torch.max(outputs, 1)
#         for t, p in zip(classes.view(-1), preds.view(-1)):
#                 confusion_matrix[t.long(), p.long()] += 1

# print(confusion_matrix)

# print(confusion_matrix.diag()/confusion_matrix.sum(1))

# import pandas as pd
# import seaborn as sn
# df_cm = pd.DataFrame(confusion_matrix.numpy(), index=[i for i in working_ds.classes],
#                      columns=[i for i in working_ds.classes])
# plt.figure(figsize=(20, 15))
# sn.heatmap(df_cm, annot=True, cmap='BuPu')

train_stats.f1, train_stats.precision, train_stats.recall = recall_precision_fmeasure(netparams, working_ds)
train_stats.confusion = confusion(netparams, working_ds).numpy().tolist()

train_stats.save(working_ds)

# shutil.move('/content/model_cifar.pt', f'/content/drive/MyDrive/results/{working_ds.name}_results/{netparams.model.name}-{datetime.now()}', copy_function=copy2)

print(train_stats.k1)
print(train_stats.k5)

"""## macierz konfuzji

### zapisz plot na dysku
"""

#  train_stats.confusion = torch.FloatTensor(train_stats.confusion)

confusion_plot = get_plot_confusion(train_stats.confusion, working_ds)

now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

dir_path = f'drive/MyDrive/results/{working_ds.name}_plots/'
file_name = f'{train_stats.model_name}_{now}'

Path(dir_path).mkdir(parents=True, exist_ok=True)

confusion_plot.figure.savefig(f"{dir_path}{file_name}.png")

"""### Visualize Sample Test Results"""

plot_test_results(netparams, working_ds)

output.eval_js('new Audio("https://assets.mixkit.co/sfx/preview/mixkit-happy-bells-notification-937.mp3").play()')

def plot_test_results2(netparams: NetParams, working_ds):
    # obtain one batch of test images
    dataiter = iter(netparams.test_loader)
    images, labels = dataiter.next()
    images.numpy()

    netparams.model.cpu()
    # get sample outputs
    output = netparams.model(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(30, 8))
    for idx in np.arange(netparams.batch_size):
        ax = fig.add_subplot(3, netparams.batch_size / 3,
                             idx + 1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[idx], (1, 2, 0)))
        ax.set_title("{}\n({})".format(working_ds.classes[preds[idx]], working_ds.classes[labels[idx]]),
                     color=("green" if preds[idx] == labels[idx].item() else "red"))

plot_test_results2(netparams, working_ds)