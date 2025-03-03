"""
AlzNetV3 Model Architecture

Translational Approach for Dementia Subtype Classification 
Using EEG Connectome Profile-Based Convolutional Neural Network

Author: T. Jungrungrueang, S. Chairat, and K. Charupanit
Affiliation: Faculty of Medicine, Prince of Songkla University
Email: thawirasm.j@psu.ac.th
Date: 28-02-2025

Model modifiers:
- in_channels: int, number of input channels (default: 4)
- out_channels: list, number of output channels for each layer (default: [256, 512, 1024])
- hidden_layer: int, size of hidden layer in fully connected layer (default: 512)
- num_classes: int, number of classification classes (default: 3)
- nm: int, number of concatenated maps in the connectome matrix (Instance-wise: 15, Feature-wise: 5, Band-wise: 3, Single: 1)

Model parameters:
- droprate: float, dropout rate (default: 0.25)

"""

import torch
import torch.nn as nn

class AlzNetV3(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: list = [256, 512, 1024], hidden_layer: int = 512, num_classes: int = 3, nm: int = 15):
        super().__init__()
        self.droprate = 0.25
        self.cnn_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels[0],kernel_size=3,stride=1,padding=2,padding_mode='circular'),
            nn.Dropout2d(p=self.droprate),
            nn.BatchNorm2d(out_channels[0]),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=out_channels[0],out_channels=out_channels[0],kernel_size=3,stride=1,padding=2,padding_mode='circular'),
            nn.Dropout2d(p=self.droprate),
            nn.BatchNorm2d(out_channels[0]),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=3),
            nn.Conv2d(in_channels=out_channels[0],out_channels=out_channels[1],kernel_size=3,stride=1,padding=2,padding_mode='circular'),
            nn.Dropout2d(p=self.droprate),
            nn.BatchNorm2d(out_channels[1]),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=out_channels[1],out_channels=out_channels[1],kernel_size=3,stride=1,padding=2,padding_mode='circular'),
            nn.Dropout2d(p=self.droprate),
            nn.BatchNorm2d(out_channels[1]),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=3),
            nn.Conv2d(in_channels=out_channels[1],out_channels=out_channels[2],kernel_size=3,stride=1,padding=2,padding_mode='circular'),
            nn.Dropout2d(p=self.droprate),
            nn.BatchNorm2d(out_channels[2]),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=out_channels[2],out_channels=out_channels[2],kernel_size=3,stride=1,padding=2,padding_mode='circular'),
            nn.Dropout2d(p=self.droprate),
            nn.BatchNorm2d(out_channels[2]),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=3),
        )
        self.fully_connected_stack = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=hidden_layer * nm),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_layer * nm,out_features=num_classes),
            nn.Softmax(dim=1),
        )
        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=out_channels[2],out_features=hidden_layer),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_layer,out_features=num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.cnn_relu_stack(x)
        # x = self.fully_connected_stack(x)
        x = self.global_pooling(x)
        return x

# Training parameters 
parameters = {
    "max_epochs": 1000,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "doEarlyStopping": True,
    "patience": 30,
    "loss_fn": nn.CrossEntropyLoss(),
}

# List of outlier subject IDs
outlier = ['02', '06', '25', '31', '50', '53', '69', '80', '86']