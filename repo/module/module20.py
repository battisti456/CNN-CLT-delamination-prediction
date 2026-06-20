import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from run.hyperparameters import Hyperparameters

depths = (64,64,128,256,512)

class Module20(nn.Module):
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        self.hp = hp
        self.transfer = smp.Linknet(
            encoder_name='resnet18',
            in_channels=1,
            encoder_depth=3,
        )
        self.encoder = self.transfer.encoder
        # self.transfer = self.Transfer(self.encoder,resnet.decoder,resnet.segmentation_head)
        self.head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(65536,64),
            nn.LeakyReLU(),
            nn.Dropout(hp.dropout),
            nn.Linear(64,32),
            nn.LeakyReLU(),
            nn.Dropout(hp.dropout),
            nn.Linear(32,hp.num_classes)
        )
    def forward(self,input):
        x = self.encoder(input['spectrogram'])[-1]
        x = self.head(x)
        return x
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
