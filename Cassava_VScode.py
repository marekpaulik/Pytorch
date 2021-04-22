#TODO
#   load data
#   create class
#   fix inference


#Load data


#Import libraries
import numpy as np 
import pandas as pd
import json
from PIL import Image
import os
import matplotlib.pyplot as plt
from collections import Counter
import math

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from efficientnet_pytorch import EfficientNet

from tqdm import tqdm

#Callbacks
# Early stopping
class EarlyStopping:
  def __init__(self, patience=3, delta=0, path='checkpoint.pt'):
    self.patience = patience
    self.delta = delta
    self.path= path
    self.counter = 0
    self.best_score = None
    self.early_stop = False

  def __call__(self, val_loss, model):
    if self.best_score is None:
      self.best_score = val_loss
      self.save_checkpoint(model)
    elif val_loss > self.best_score:
      self.counter +=1
      if self.counter >= self.patience:
        self.early_stop = True 
    else:
      self.best_score = val_loss
      self.save_checkpoint(model)
      self.counter = 0      

  def save_checkpoint(self, model):
    torch.save(model.state_dict(), self.path)

#Create dataloaders
def load_data(data_dir, Transform, sample, batch_size):
  ########################################################################################
# data_dir - directory with images in subfolders, subfolders name are categories
# Transform - data augmentations
# sample - if the dataset is imbalanced set to true and RandomWeightedSampler will be used
  #########################################################################################
  train_full = torchvision.datasets.ImageFolder(data_dir, transform=Transform)
  train_set, val_set = random_split(train_full, [math.floor(len(train_full)*0.8), math.ceil(len(train_full)*0.2)])

  train_classes = [label for _, label in train_set]
  if sample:
    # Need to get weight for every image in the dataset
    class_count = Counter(train_classes)
    class_weights = torch.Tensor([len(train_classes)/c for c in pd.Series(class_count).sort_index().values]) # Cant iterate over class_count because dictionary is unordered

    sample_weights = [0] * len(train_set)
    for idx, (image, label) in enumerate(train_set):
      class_weight = class_weights[label]
      sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples = len(train_set), replacement=True)  
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
  else:
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

  val_loader = DataLoader(val_set, batch_size=batch_size)

  return train_loader, val_loader, train_classes    


  # Load model, criterion, optimizer
  def load_model(arch, num_classes, lr, loss_weights, freeze_backbone, device, train_classes):
  ########################################################################################
# arch - choose the pretrained architecture from resnet or efficientnetb7
# loss_weights - if the dataset is imbalanced set to true and weight parameter will be passed to loss function
# freeze_backbone - if using pretrained architecture freeze all but the classification layer
# train_classes - helper parameter passed from load_data(), needed if loss_weights=True
  #########################################################################################  
  if arch == 'resnet':
    model = torchvision.models.resnet50(pretrained=True)
    if freeze_backbone:
      for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
  elif arch == 'efficient-net':
    model = EfficientNet.from_pretrained('efficientnet-b7')
    if freeze_backbone:
      for param in model.parameters():
        param.requires_grad = False
    model._fc = nn.Linear(in_features=model._fc.in_features, out_features=num_classes)    

  model = model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr) 

  if loss_weights:
    class_count = Counter(train_classes)
    class_weights = torch.Tensor([len(train_classes)/c for c in pd.Series(class_count).sort_index().values]) # Cant iterate over class_count because dictionary is unordered
    class_weights = class_weights.to(device)  
    criterion = nn.CrossEntropyLoss(class_weights)
  else:
    criterion = nn.CrossEntropyLoss() 
  
  return model, optimizer, criterion   

  # Train function
  def train(model, criterion, optimizer, num_epochs, use_amp, freeze_backbone,
          unfreeze_after, tensorboard, stop_early, device, train_loader, val_loader):
  if tensorboard:
    writer = SummaryWriter('runs/sampler_cassava') #TODO parametrize
  if stop_early:
    early_stopping = EarlyStopping(
    patience=5, 
    path='/content/drive/My Drive/ColabNotebooks/Cassava/sampler_checkpoint.pt') #TODO parametrize
  
  num_epochs = num_epochs
  step_train = 0
  step_val = 0

  scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

  for epoch in range(num_epochs):
    if freeze_backbone:
      if epoch == unfreeze_after:  # Unfreeze after x epochs
        for param in model.parameters():
          param.requires_grad = True

    train_loss = list() # Every epoch check average loss per batch 
    train_acc = list()
    model.train()
    for i, (images, targets) in enumerate(tqdm(train_loader)):
      images = images.to(device)
      targets = targets.to(device)

      with torch.cuda.amp.autocast(enabled=use_amp): #mixed precision
        logits = model(images)
        loss = criterion(logits, targets)

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      optimizer.zero_grad()

      train_loss.append(loss.item())

      #Calculate running train accuracy
      predictions = torch.argmax(logits, dim=1)
      num_correct = sum(predictions.eq(targets))
      running_train_acc = float(num_correct) / float(images.shape[0])
      train_acc.append(running_train_acc)

      # Plot to tensorboard
      if tensorboard:
        img_grid = torchvision.utils.make_grid(images[:10])
        writer.add_image('Cassava_images', img_grid) # Check how transformed images look in training
        #writer.add_histogram('fc', model.fc.weight) # Check if our weights change during trianing

        writer.add_scalar('training_loss', loss, global_step=step_train)
        writer.add_scalar('training_acc', running_train_acc, global_step=step_train)
        step_train +=1

    print(f'Epoch {epoch}/{num_epochs-1}')  
    print(f'Training loss: {torch.tensor(train_loss).mean():.2f}') 

    val_losses = list()
    val_accs = list()
    model.eval()
    with torch.no_grad():
      for (images, targets) in val_loader:
        images = images.to(device)
        targets = targets.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
          logits = model(images)
          loss = criterion(logits, targets)
        val_losses.append(loss.item())      
        
        predictions = torch.argmax(logits, dim=1)
        num_correct = sum(predictions.eq(targets))
        running_val_acc = float(num_correct) / float(images.shape[0])

        val_accs.append(running_val_acc)
        
        if tensorboard:
          writer.add_scalar('validation_loss', loss, global_step=step_val)
          writer.add_scalar('validation_acc', running_val_acc, global_step=step_val)
          step_val +=1

      val_loss = torch.tensor(val_losses).mean()
      val_acc = torch.tensor(val_accs).mean() # Average acc per batch
      
      print(f'Validation loss: {val_loss:.2f}')  
      print(f'Validation accuracy: {val_acc:.2f}') 
      
      if stop_early:
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
          print('Early Stopping')
          print(f'Best validation loss: {early_stopping.best_score}')
          break


# Main function

def main(num_classes, device, data_dir='train/train', Transform=None, sample=False, loss_weights=False,
         batch_size=64, lr=1e-4, arch='resnet', tensorboard=True,
         stop_early=True, use_amp=True, freeze_backbone=True, num_epochs=10, unfreeze_after=5):
  
  train_loader, val_loader, train_classes = load_data(data_dir, Transform=Transform,
                                       sample=sample, batch_size=batch_size)
  
  model, optimizer, criterion = load_model(arch=arch, num_classes=num_classes,
                                lr=lr, loss_weights=loss_weights,
                                freeze_backbone=freeze_backbone, device=device, train_classes=train_classes)

  train(model=model, optimizer=optimizer, criterion=criterion, num_epochs=num_epochs, freeze_backbone=freeze_backbone,
        unfreeze_after=unfreeze_after, tensorboard=tensorboard,
        stop_early=stop_early, use_amp=use_amp, device=device, train_loader=train_loader,
        val_loader=val_loader)
  


# Training
Transform = T.Compose(
    [T.ToTensor(),
     T.Resize((256, 256)),
     T.RandomRotation(90),
     T.RandomHorizontalFlip(p=0.5),
     T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

main(arch = 'efficient-net', Transform=Transform, sample=True, num_classes=5, device=device, num_epochs=20, batch_size=16)