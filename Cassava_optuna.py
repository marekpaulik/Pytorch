#Import libraries
import numpy as np 
import pandas as pd
import json
from PIL import Image
import os
import matplotlib.pyplot as plt
from collections import Counter
import math
import random

import optuna
from optuna.trial import TrialState


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from efficientnet_pytorch import EfficientNet

from tqdm import tqdm

torch.cuda.empty_cache()


cfg = {
    'arch': 'resnet',
    'batch_size': 128,
    'data_dir': "C:/Users/marek/Deep Learning/Data/cassava-disease/train/train" ,
    'num_classes': 5,
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    'sample': True,
    'loss_weights': False,
    'tensorboard': True,
    'stop_early': True,
    'patience': 5,
    'use_amp': True,
    'freeze_backbone': True,
    'unfreeze_after': 5,
    'num_epochs': 3,
    'n_trials': 6,
    'n_startup_trials': 1,
    'n_warmup_steps': 3,
    'interval_steps': 1,
    'seed': 42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


#Callbacks
# Early stopping
class EarlyStopping:
  def __init__(self, patience=3, delta=0, path='checkpoint.pt'):
    self.patience = patience
    self.delta = delta
    self.path = path
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
    def __init__(self, data_dir, num_classes, device, Transform=None, sample=False, loss_weights=cfg['loss_weights'], batch_size=cfg['batch_size'],
     lr=1e-4, tensorboard=cfg['tensorboard'], stop_early=cfg['stop_early'], use_amp=cfg['use_amp'], freeze_backbone=cfg['freeze_backbone']):
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


    def load_model(self, arch=cfg['arch']):
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

        #self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr) 

        if self.loss_weights:
            class_count = Counter(self.train_classes)
            class_weights = torch.Tensor([len(self.train_classes)/c for c in pd.Series(class_count).sort_index().values]) # Cant iterate over class_count because dictionary is unordered
            class_weights = class_weights.to(self.device)  
            self.criterion = nn.CrossEntropyLoss(class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss() 
        
        #return model, optimizer, criterion  

    def fit_one_epoch(self, optimizer, train_loader, scaler, epoch, num_epochs): 
        step_train = 0

        train_losses = list() # Every epoch check average loss per batch 
        train_acc = list()
        self.model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, targets) in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp): #mixed precision
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

            train_losses.append(loss.item())

            #Calculate running train accuracy
            predictions = torch.argmax(logits, dim=1)
            num_correct = sum(predictions.eq(targets))
            running_train_acc = float(num_correct) / float(images.shape[0])
            train_acc.append(running_train_acc)

            # Plot to tensorboard
            if self.tensorboard:
                img_grid = torchvision.utils.make_grid(images[:10])
                self.writer.add_image('Cassava_images', img_grid) # Check how transformed images look in training
                #writer.add_histogram('fc', model.fc.weight) # Check if our weights change during trianing

                self.writer.add_scalar('training_loss', loss, global_step=step_train)
                self.writer.add_scalar('training_acc', running_train_acc, global_step=step_train)
                step_train +=1
        train_loss = torch.tensor(train_losses).mean()
        print(f'Epoch {epoch}/{num_epochs-1}')  
        print(f'Training loss: {train_loss:.2f}')

    def val_one_epoch(self, val_loader, scaler):
        val_losses = list()
        val_accs = list()
        self.model.eval()
        step_val = 0
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
                    self.writer.add_scalar('validation_loss', loss, global_step=step_val)
                    self.writer.add_scalar('validation_acc', running_val_acc, global_step=step_val)
                    step_val +=1

            self.val_loss = torch.tensor(val_losses).mean()
            val_acc = torch.tensor(val_accs).mean() # Average acc per batch
        
            print(f'Validation loss: {self.val_loss:.2f}')  
            print(f'Validation accuracy: {val_acc:.2f}') 


    def __call__(self, trial, num_epochs=cfg['num_epochs'], unfreeze_after=cfg['unfreeze_after'], checkpoint_dir='checkpoint.pt'):
        self.load_model()

        if self.tensorboard:
            self.writer = SummaryWriter('runs/sampler_cassava')

        if self.stop_early:
             early_stopping = EarlyStopping(
             patience=cfg['patience'], 
             path=checkpoint_dir)    

        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=lr)

        train_loader, val_loader = self.load_data()

        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) 


        for epoch in range(num_epochs):
            if self.freeze_backbone:
                if epoch == unfreeze_after:  # Unfreeze after x epochs
                    for param in self.model.parameters():
                        param.requires_grad = True
            self.fit_one_epoch(optimizer, train_loader, scaler, epoch, num_epochs)
            self.val_one_epoch(val_loader, scaler)

            if self.stop_early:
                 early_stopping(self.val_loss, self.model)
                 if early_stopping.early_stop:
                     print('Early Stopping')
                     print(f'Best validation loss: {early_stopping.best_score}')
                     break

            trial.report(self.val_loss, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return self.val_loss                
                    


Transform = T.Compose(
                    [T.ToTensor(),
                     T.Resize((256, 256)),
                     T.RandomRotation(90),
                     T.RandomHorizontalFlip(p=0.5),
                     T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

seed_everything(cfg['seed'])

classifier = CassavaClassifier(data_dir=cfg['data_dir'], num_classes=cfg['num_classes'], device=cfg['device'],
 sample=cfg['sample'], Transform=Transform)


study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(
        n_startup_trials=cfg['n_startup_trials'], n_warmup_steps=cfg['n_warmup_steps'], interval_steps=cfg['interval_steps']))
study.optimize(classifier, n_trials=cfg['n_trials'])

complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)


print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

df = study.trials_dataframe()


