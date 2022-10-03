from dataset_loader import data_loader
from model import TUMORCLASSIFIER
from utility import accuracy, evaluate_proposed_model
import config

import torch
import warnings
import torch.optim as optim
from tqdm import tqdm
import sys
import numpy as np
import os
import pandas as pd


import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device='cpu'
def trainer():

    train_loader, validation_loader, test_loader = data_loader()
    
    model = TUMORCLASSIFIER()
    model = model.to(device)
    
    error = torch.nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNINIG_RATE)  
    
    val_acces = []
    val_losses = []
    
    train_losses = []
    train_acces = []
    
    epoch_train_loss = []
    epoch_train_acc = []
    
    epoch_val_loss = []
    epoch_val_Acc = []

    #early_stopping =
    print('*************** Model Training Started ************** ')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for epoch in range(config.EPOCHS):
            total_correct = 0
            train_loss = 0
            
            model.train()           
            train_bar = tqdm(train_loader, file=sys.stdout)
            
            for step, (samples, labels) in enumerate(train_bar):
                #pdb.set_trace()
                
                # Transfering samples and labels to GPU if available
                samples, labels = samples.to(device), labels.to(device)
                labels = labels.type(torch.int64)

                # Forward pass
                outputs = model(samples)
                
                
                loss = error(outputs, labels)

                # Initializing a gradient as 0 so there is no mixing of gradient among the batches
                optimizer.zero_grad()

                # Propagating the error backward
                loss.backward()

                # Optimizing the parameters
                optimizer.step()

                train_losses.append(loss.item())
                train_acces.append(accuracy(outputs, labels))
                #pdb.set_trace()
            
            #pdb.set_trace()
            model.eval()
            validation_bar = tqdm(validation_loader, file=sys.stdout)
            #print('hi')
            for step, (samples, labels) in enumerate(validation_bar):

                samples, labels = samples.to(device), labels.to(device)
                outputs = model(samples)
                labels = labels.type(torch.int64)
                loss = error(outputs, labels)
                val_losses.append(loss.item())
                val_acces.append(accuracy(outputs, labels))


           
            #, train_acc, val_loss, val_acc)
            train_loss = round(np.average(train_losses),4)
            train_acc = round(np.average(train_acces), 4)
            val_loss = round(np.average(val_losses), 4)
            val_acc = round(np.average(val_acces), 4)
            

            
            epoch_train_loss.append(train_loss)
            epoch_train_acc.append(train_acc)

            epoch_val_loss.append(val_loss)
            epoch_val_Acc.append(val_acc)
            
            
            
            
            epoch += 1

            # Printing the model Traning Accuracy and Testing Accuracy

            print("Epoch: {}, Train Loss: {}, Val Loss: {}, Train Accuracy: {}, Val Accuracy: {}".format(epoch, train_loss, val_loss, train_acc, val_acc))
        
        train_val_metrics = {'Train Accuracy':epoch_train_acc ,'Train Loss':epoch_train_loss, 'Validation Accuracy':epoch_val_Acc , 'Validation Loss':epoch_val_loss}
        
        train_val_metrics = pd.DataFrame(train_val_metrics)
        train_val_metrics.to_csv(os.getcwd()+'/Data/train_val_metrics.csv')
       
        print('*************** Model Training Finished ************** ')
        print('*************** Testing Model on the Test Data ************** ')

        
        # evaluate_proposed_model(model, test_loader)

        print('*************** Saving the Trained Model ************** ')

        path_to_saved_model = os.getcwd() +  '/PretrainedModel/'

        # checking if directory exists if not create one
        if not os.path.exists(path_to_saved_model):
            os.mkdir(path_to_saved_model)

        #assigning the model name accoring to train times
        version = 1
        while(True):

            name = 'modelv'+str(version) + '.pth'
            model_name = path_to_saved_model +  name
            if os.path.exists(model_name):
                version += 1
            else:
                break


        torch.save(model.state_dict(),  model_name)

        print('*************** Model Saved Sucessfully ************** ')
    
    
    
trainer()
    
    
    
    
    