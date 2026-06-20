import segmentation_models_pytorch as smp
import rff.layers
import rff.functional
import torch
import torch.nn as nn
import torch.nn.functional as F
from run.hyperparameters import Hyperparameters
import warnings

depths = (64,64,128,256,512)

class Module21(nn.Module):
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        self.hp = hp
        self.transfer = smp.Linknet(
            encoder_name='resnet18',
            in_channels=1,
            encoder_depth=3,
        )
        self.encoder = self.transfer.encoder
        reduced_size = (8,4)
        self._rff_encoded_size = 128
        self._rff_input_size = 128*reduced_size[0]*reduced_size[1]
        self.flatten_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(reduced_size),
            nn.Flatten(1),
        )
        self._set_sigma = False
        if hp.module_hyperparameters['sigma'] is None:
            self._set_sigma = True
        self.rff = rff.layers.GaussianEncoding(
                sigma = 1 if self._set_sigma else hp.module_hyperparameters['sigma'],
                input_size=self._rff_input_size,
                encoded_size=self._rff_encoded_size
            )
        self.linear = nn.Sequential(
            nn.Linear(self._rff_encoded_size*2,64),
            nn.LeakyReLU(),
            nn.Dropout(hp.dropout),
            nn.Linear(64,32),
            nn.LeakyReLU(),
            nn.Dropout(hp.dropout),
            nn.Linear(32,hp.num_classes)
        )
    def forward(self,input):
        x = self.encoder(input['spectrogram'])[-1]
        x = self.flatten_encoder(x)
        if self._set_sigma and self.training:
            with torch.no_grad():
                num_samples = min(x.size(0), 256)
                idx = torch.randperm(x.shape[0])[:num_samples]
                d = torch.cdist(x[idx], x[idx])
                with warnings.catch_warnings(action='ignore'):
                    #torch.median not supported on GPU, fine since it only happens once per fold
                    sigma:float =  torch.median(d).item()
                b = rff.functional.sample_b(sigma,(self._rff_encoded_size,self._rff_input_size))
                b = b.to(self.rff.b.device)
                self.rff.b.copy_(b)
            self._set_sigma = False
        x = self.rff(x)
        x = self.linear(x)
        return x
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
