import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        out = out + x
        out = self.final_relu(out)
        return out
    
set_seed(42)
class SatelliteCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer= nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  #3 input channels, 32 output channels
            nn.BatchNorm2d(32),#Batch Normalization
            nn.ReLU(),#non-linearity
            nn.MaxPool2d(kernel_size=2, stride=2),#downsampling
            ResidualBlock(32),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), #32 input channels, 64 output channels
            nn.BatchNorm2d(32),#Batch Normalization
            nn.ReLU(),#non-linearity
            ResidualBlock(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), #32 input channels, 64 output channels
            nn.BatchNorm2d(64),#Batch Normalization
            nn.ReLU(),#non-linearity
            nn.MaxPool2d(kernel_size=2, stride=2),#downsampling
            ResidualBlock(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(128),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128)
        )
        self.flatten = nn.Flatten()#flatten the output
        self.linear_layer = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),# transform the flattened image to 256 features
            nn.ReLU(),# non-linearity
            nn.Dropout(0.45),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(64, 10),# 64 features to 10 classes (output layer)
        )
    def forward(self,x):
        x=self.conv_layer(x)
        x=self.flatten(x)
        x=self.linear_layer(x)
        return x
    