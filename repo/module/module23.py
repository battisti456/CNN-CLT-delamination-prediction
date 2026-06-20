import segmentation_models_pytorch as smp
import rff.layers
import rff.functional
import torch
import torch.nn as nn
import torch.nn.functional as F
from run.hyperparameters import Hyperparameters
import warnings


class Module23(nn.Module):
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        self.hp = hp
        self.transfer = smp.Linknet(
            encoder_name='resnet18',
            in_channels=1,
            encoder_depth=3,
        )
        self.encoder = self.transfer.encoder
        reduced_size = (8,8)
        p1 = self.encoder.out_channels[-1]
        p2 = int(p1/2)
        p3 = int(p2/2)
        self.flatten_encoder = nn.Sequential(
            nn.Conv2d(p1,p2,1),
            nn.Dropout2d(hp.dropout),
            nn.BatchNorm2d(p2),
            nn.GELU(),
            nn.Conv2d(p2,p3,1),
            nn.Dropout2d(hp.dropout),
            nn.BatchNorm2d(p3),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(reduced_size),
            nn.Flatten(1),
            nn.LayerNorm(p3*reduced_size[0]*reduced_size[1]),
            nn.Dropout(hp.dropout)
        )
        self.linear = nn.Linear(p3*reduced_size[0]*reduced_size[1],hp.num_classes)
    def forward(self,input):
        x = self.encoder(input['spectrogram'])[-1]
        x = self.flatten_encoder(x)
        x = self.linear(x)
        return x
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
