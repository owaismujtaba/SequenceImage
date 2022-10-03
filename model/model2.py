from torch import nn
import config
import pdb
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.nn import LSTM
import torch

class TUMORCLASSIFIER(nn.Module):
    
    
    def __init__(self):
        super(TUMORCLASSIFIER, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(64)
            #nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=1),
            nn.ReLU(),
            #nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, padding=1),
            nn.ReLU(),
            #nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(256),
            #nn.MaxPool2d(2)
        )
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(128),
            #nn.MaxPool2d(2)
        )
        
        
        self.fc1 = nn.Linear(15488, 1024)
        self.lstm1 = LSTM(11*11, 128)
        self.lstm2 = LSTM(128, 128)
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, config.NUMBER_OF_CLASSES)
        
    
    
    
    def forward(self, x):
        size = x.shape[0]
        x = x.unsqueeze(1)
        #print(x.shape)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        #print(out.shape)
        lstm_input = out.reshape(size, 128, -1)
        
        lstm1 = self.lstm1(lstm_input)[0]
        lstm2 = self.lstm2(lstm1)[0]
        lstm2 = lstm2.reshape(size, -1)
        
        
        
        
        
        #pdb.set_trace()
        out = out.view(size, -1)
        conc = torch.cat((lstm2, out), 1)
        out = self.fc1(out)
        
        out = self.fc2(out)
        out = self.fc3(out)
        #print(out.shape)
        
        return out
    
    
        
class TUMORCLASSIFIER1(nn.Module):
    
    
    def __init__(self):
        super(TUMORCLASSIFIER, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(64)
            #nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=1),
            nn.Sigmoid(),
            #nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, padding=1),
            nn.Sigmoid(),
            #nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            #nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(256),
            #nn.MaxPool2d(2)
        )
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.BatchNorm2d(128),
            #nn.MaxPool2d(2)
        )
        
        
        self.fc1 = nn.Linear(15488, 1024)
        self.lstm1 = LSTM(11*11, 128)
        self.lstm2 = LSTM(128, 128)
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, config.NUMBER_OF_CLASSES)
        
    
    
    
    def forward(self, x):
        size = x.shape[0]
        x = x.unsqueeze(1)
        #print(x.shape)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        #print(out.shape)
        lstm_input = out.reshape(size, 128, -1)
        
        lstm1 = self.lstm1(lstm_input)[0]
        lstm2 = self.lstm2(lstm1)[0]
        lstm2 = lstm2.reshape(size, -1)
        
        
        
        
        
        #pdb.set_trace()
        out = out.view(size, -1)
        conc = torch.cat((lstm2, out), 1)
        out = self.fc1(out)
        
        out = self.fc2(out)
        out = self.fc3(out)
        #print(out.shape)
        
        return out
    