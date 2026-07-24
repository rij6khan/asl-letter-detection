import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data import Dataset

import torchvision.datasets as dset
import torchvision.transforms as T

# for plotting
import matplotlib.pyplot as plt


arr = []
with open('archive/sign_mnist_train.csv', mode='r') as file:
    reader = csv.DictReader(file) 
    for row in reader:
        label = int(row['label'])
        pixels = [int(v) for k, v in row.items() if k != 'label']
        matrix_28x28 = np.array(pixels).reshape(28, 28)
        sample_pair = (label, matrix_28x28)
        arr.append(sample_pair)
arrTwo = []
with open('archive/sign_mnist_test.csv', mode='r') as file:
    reader = csv.DictReader(file) 
    for row in reader:
        label = int(row['label'])
        pixels = [int(v) for k, v in row.items() if k != 'label']
        matrix_28x28 = np.array(pixels).reshape(28, 28)
        sample_pair = (label, matrix_28x28)
        arrTwo.append(sample_pair)

def flatten(x, start_dim=1, end_dim=-1):
  return x.flatten(start_dim=start_dim, end_dim=end_dim)

class Flatten(nn.Module):
  def forward(self, x):
    return flatten(x)

class SignMNISTDataset(Dataset):
    def __init__(self, data_list, transform=None, train = True):
        self.data = data_list
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, image = self.data[idx]
        image = image.astype(np.uint8)
        if self.transform:
            image = self.transform(image)
        return image, label


transform = T.Compose([
    T.ToPILImage(), 
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)) 
])

full_dataset = SignMNISTDataset(arr, transform=transform, train = True)
test_dataset = SignMNISTDataset(arrTwo, transform=transform, train = False)
num_samples = len(full_dataset)
num_train = int(0.8 * num_samples)

loader_train = DataLoader(
    full_dataset, 
    batch_size=64, 
    sampler=torch.utils.data.SubsetRandomSampler(range(num_train))
)

loader_val = DataLoader(
    full_dataset, 
    batch_size=64, 
    sampler=torch.utils.data.SubsetRandomSampler(range(num_train, num_samples))
)

loader_test = DataLoader(    
    test_dataset, 
    batch_size=64
    )

dtype = torch.float
ltype = torch.long
#Change it to gpu if you own a NVIDIA 
device = torch.device('cpu')
print_every = 100

def check_accuracy_part34(loader, model):
  if loader.dataset.train:
    print('Checking accuracy on validation set')
  else:
    print('Checking accuracy on test set')   
  num_correct = 0
  num_samples = 0
  model.eval() 
  with torch.no_grad():
    for x, y in loader:
      x = x.to(device=device, dtype=dtype) 
      y = y.to(device=device, dtype=ltype)
      scores = model(x)
      _, preds = scores.max(1)
      num_correct += (preds == y).sum()
      num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
  return acc

def adjust_learning_rate(optimizer, lrd, epoch, schedule):
  if epoch in schedule:
    for param_group in optimizer.param_groups:
      print('lr decay from {} to {}'.format(param_group['lr'], param_group['lr'] * lrd))
      param_group['lr'] *= lrd

def train_part345(model, optimizer, epochs=2, learning_rate_decay=.1, schedule=[], verbose=True):
  model = model.to(device=device) 
  num_iters = epochs * len(loader_train)
  if verbose:
    num_prints = num_iters
  else:
    num_prints = epochs
  acc_history = torch.zeros(num_prints, dtype=torch.float)
  iter_history = torch.zeros(num_prints, dtype=torch.long)
  for e in range(epochs):
    
    adjust_learning_rate(optimizer, learning_rate_decay, e, schedule)
    
    for t, (x, y) in enumerate(loader_train):
      model.train()
      x = x.to(device=device, dtype=dtype)  
      y = y.to(device=device, dtype=ltype)

      scores = model(x)
      loss = F.cross_entropy(scores, y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      tt = t + e * len(loader_train)

      if verbose and (tt % print_every == 0 or (e == epochs-1 and t == len(loader_train)-1)):
        print('Epoch %d, Iteration %d, loss = %.4f' % (e, tt, loss.item()))
        acc = check_accuracy_part34(loader_val, model)
        acc_history[tt // print_every] = acc
        iter_history[tt // print_every] = tt
        print()
      elif not verbose and (t == len(loader_train)-1):
        print('Epoch %d, Iteration %d, loss = %.4f' % (e, tt, loss.item()))
        acc = check_accuracy_part34(loader_val, model)
        acc_history[e] = acc
        iter_history[e] = tt
        print()
  return acc_history, iter_history

C, H, W = 1, 28, 28 
num_classes = 25     

channel_1 = 32
channel_2 = 16
kernel_size_1 = 5
pad_size_1 = 2
kernel_size_2 = 3
pad_size_2 = 1

learning_rate = 1e-2
momentum = 0.5
weight_decay = 1e-4 

model = nn.Sequential(OrderedDict([
  ('conv1', nn.Conv2d(C, channel_1, kernel_size_1, padding=pad_size_1)),
  ('relu1', nn.ReLU()),
  ('conv2', nn.Conv2d(channel_1, channel_2, kernel_size_2, padding=pad_size_2)),
  ('relu2', nn.ReLU()),
  
  ('flatten', Flatten()),

  ('fc3', nn.Linear(channel_2 * H * W, num_classes)),
]))

optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                      weight_decay=weight_decay, momentum=momentum, nesterov=False)

three_layer_conv_seq_acc_history, _ = train_part345(model, optimizer)

best_model = model
acc_final = check_accuracy_part34(loader_test, best_model)