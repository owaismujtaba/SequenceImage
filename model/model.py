from torch import nn
import config
import pdb
import torch.nn.functional as F


class TUMORCLASSIFIER(nn.Module):
    
    
    def __init__(self):
        super(TUMORCLASSIFIER, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25, inplace=False)
            #nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25, inplace=False),
            nn.MaxPool2d(2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25, inplace=False),
            nn.MaxPool2d(2)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25, inplace=False),
            nn.MaxPool2d(2)
        )
        
        
        self.fc1 = nn.Linear(64*12*12, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, config.NUMBER_OF_CLASSES)
        
    
    
    
    def forward(self, x):
        
        size = x.shape[0]
        x = x.unsqueeze(1)
        
        out = self.layer1(x)
        
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = out.view(size, -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out
    
    
        