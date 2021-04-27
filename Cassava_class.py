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

torch.cuda.empty_cache()

#Callbacks
# Early stopping
class EarlyStopping:
  def __init__(self, patience=1, delta=0, path='checkpoint.pt'):
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


class CassavaClassifier():
    def __init__(self, data_dir, num_classes, device, Transform=None, sample=False, loss_weights=False, batch_size=16,
     lr=1e-4, tensorboard=True, stop_early=True, use_amp=True, freeze_backbone=True):
    ########################################################################################
    # data_dir - directory with images in subfolders, subfolders name are categories
    # Transform - data augmentations
    # sample - if the dataset is imbalanced set to true and RandomWeightedSampler will be used
    # loss_weights - if the dataset is imbalanced set to true and weight parameter will be passed to loss function
    # freeze_backbone - if using pretrained architecture freeze all but the classification layer
    #########################################################################################
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.device = device
        self.sample = sample
        self.loss_weights = loss_weights
        self.batch_size = batch_size
        self.lr = lr
        self.tensorboard = tensorboard
        self.stop_early = stop_early
        self.use_amp = use_amp
        self.freeze_backbone = freeze_backbone
        self.Transform = Transform

    def load_data(self):
        train_full = torchvision.datasets.ImageFolder(self.data_dir, transform=self.Transform)
        train_set, val_set = random_split(train_full, [math.floor(len(train_full)*0.8), math.ceil(len(train_full)*0.2)])

        self.train_classes = [label for _, label in train_set]
        if self.sample:
            # Need to get weight for every image in the dataset
            class_count = Counter(self.train_classes)
            class_weights = torch.Tensor([len(self.train_classes)/c for c in pd.Series(class_count).sort_index().values]) # Cant iterate over class_count because dictionary is unordered

            sample_weights = [0] * len(train_set)
            for idx, (image, label) in enumerate(train_set):
                class_weight = class_weights[label]
                sample_weights[idx] = class_weight

            sampler = WeightedRandomSampler(weights=sample_weights,
                                            num_samples = len(train_set), replacement=True)  
            train_loader = DataLoader(train_set, batch_size=self.batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        val_loader = DataLoader(val_set, batch_size=self.batch_size)

        return train_loader, val_loader


    def load_model(self, arch='resnet'):
        ##############################################################################################################
        # arch - choose the pretrained architecture from resnet or efficientnetb7
        ############################################################################################################## 
        if arch == 'resnet':
            self.model = torchvision.models.resnet50(pretrained=True)
            if self.freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
            self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.num_classes)
        elif arch == 'efficient-net':
            self.model = EfficientNet.from_pretrained('efficientnet-b7')
            if self.freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
            self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=self.num_classes)    

        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr) 

        if self.loss_weights:
            class_count = Counter(self.train_classes)
            class_weights = torch.Tensor([len(self.train_classes)/c for c in pd.Series(class_count).sort_index().values]) # Cant iterate over class_count because dictionary is unordered
            class_weights = class_weights.to(self.device)  
            self.criterion = nn.CrossEntropyLoss(class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss() 
        
        #return model, optimizer, criterion  


    def fit(self, train_loader, val_loader, num_epochs=10, unfreeze_after=5, checkpoint_dir='sampler_checkpoint'):
        ##############################################################################################################
        # unfreeze_after - unfreeze the backbone of pretrained model after x epochs
        ############################################################################################################## 
        if self.tensorboard:
            writer = SummaryWriter('runs/sampler_cassava') #TODO parametrize
        if self.stop_early:
            early_stopping = EarlyStopping(
            patience=5, 
            path=checkpoint_dir)
        
        step_train = 0
        step_val = 0

        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        for epoch in range(num_epochs):
            if self.freeze_backbone:
                if epoch == unfreeze_after:  # Unfreeze after x epochs
                    for param in self.model.parameters():
                        param.requires_grad = True

            train_loss = list() # Every epoch check average loss per batch 
            train_acc = list()
            self.model.train()
            for i, (images, targets) in enumerate(tqdm(train_loader)):
                images = images.to(self.device)
                targets = targets.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp): #mixed precision
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                self.optimizer.zero_grad()

                train_loss.append(loss.item())

                #Calculate running train accuracy
                predictions = torch.argmax(logits, dim=1)
                num_correct = sum(predictions.eq(targets))
                running_train_acc = float(num_correct) / float(images.shape[0])
                train_acc.append(running_train_acc)

                # Plot to tensorboard
                if self.tensorboard:
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
            self.model.eval()
            with torch.no_grad():
                for (images, targets) in val_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        logits = self.model(images)
                        loss = self.criterion(logits, targets)
                    val_losses.append(loss.item())      
                
                    predictions = torch.argmax(logits, dim=1)
                    num_correct = sum(predictions.eq(targets))
                    running_val_acc = float(num_correct) / float(images.shape[0])

                    val_accs.append(running_val_acc)
                
                    if self.tensorboard:
                        writer.add_scalar('validation_loss', loss, global_step=step_val)
                        writer.add_scalar('validation_acc', running_val_acc, global_step=step_val)
                        step_val +=1

                val_loss = torch.tensor(val_losses).mean()
                val_acc = torch.tensor(val_accs).mean() # Average acc per batch
            
                print(f'Validation loss: {val_loss:.2f}')  
                print(f'Validation accuracy: {val_acc:.2f}') 
            
                if self.stop_early:
                    early_stopping(val_loss, self.model)
                    if early_stopping.early_stop:
                        print('Early Stopping')
                        print(f'Best validation loss: {early_stopping.best_score}')
                        break



# Run
Transform = T.Compose(
                    [T.ToTensor(),
                    T.Resize((256, 256)),
                    T.RandomRotation(90),
                    T.RandomHorizontalFlip(p=0.5),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_dir = "C:/Users/marek/Deep Learning/Data/cassava-disease/train/train"

classifier = CassavaClassifier(data_dir=data_dir, num_classes=5, device=device, sample=True, Transform=Transform)
train_loader, val_loader = classifier.load_data()
classifier.load_model()
classifier.fit(num_epochs=2, unfreeze_after=1, train_loader=train_loader, val_loader=val_loader)

# Inference
model = torchvision.models.resnet50()
#model = EfficientNet.from_name('efficientnet-b7')
model._fc = nn.Linear(in_features=model._fc.in_features, out_features=5)
model = model.to(device)
checkpoint = torch.load('C:/Users/marek/Deep Learning/Data/cassava-disease/sampler_checkpoint.pt')
model.load_state_dict(checkpoint)
model.eval()


# Dataset for test data
class Cassava_Test(Dataset):
  def __init__(self, dir, transform=None):
    self.dir = dir
    self.transform = transform

    self.images = os.listdir(self.dir)  

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    img = Image.open(os.path.join(self.dir, self.images[idx]))
    return self.transform(img), self.images[idx] 


test_dir = 'C:/Users/marek/Deep Learning/Data/cassava-disease/test/test/0'
test_set = Cassava_Test(test_dir, transform=Transform)
test_loader = DataLoader(test_set, batch_size=4)  

# Test loop
sub = pd.DataFrame(columns=['category', 'id'])
id_list = []
pred_list = []

model = model.to(device)

with torch.no_grad():
  for (image, image_id) in test_loader:
    image = image.to(device)

    logits = model(image)
    predicted = list(torch.argmax(logits, 1).cpu().numpy())

    for id in image_id:
      id_list.append(id)
  
    for prediction in predicted:
      pred_list.append(prediction)
sub['category'] = pred_list
sub['id'] = id_list

mapping = {0:'cbb', 1:'cbsd', 2:'cgm', 3:'cmd', 4:'healthy'}

sub['category'] = sub['category'].map(mapping)
sub = sub.sort_values(by='id')

sub.to_csv('Cassava_sub.csv', index=False)