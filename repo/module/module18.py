import torch.nn as nn
import torch
from torch.nn import _reduction as _Reduction, functional as F
from run.hyperparameters import Hyperparameters
from module.module13 import Stack

class Stack_Sequence_Transpose(nn.Sequential):
    def __init__(
                self,
                k:int,
                num_layers:int,
                in_depth:int,
                out_depth:int,
                stride:int
        ):
            super().__init__()
            for i in range(num_layers):
                self.append(nn.BatchNorm1d(out_depth))
                self.append(nn.LeakyReLU(inplace=True))
                self.append(
                    nn.ConvTranspose1d(
                        out_depth,
                        in_depth if i==num_layers-1 else out_depth,
                        k,
                        stride=stride if i == num_layers-1 else 1,
                        padding=2,
                        bias=False
                    )
                )

class Stack_Transpose(nn.Module):
    def __init__(
            self,
            k:int,
            num_layers:int,
            in_depth:int,
            out_depth:int,
            stride:int
    ):
        super().__init__()
        self.sequence = Stack_Sequence_Transpose(k,num_layers,in_depth,out_depth,stride)
        self.gate = nn.ConvTranspose1d(out_depth,in_depth,1,stride=stride,bias=False)
        self.res_cat_norm = nn.BatchNorm1d(out_depth)
        self.res_comb_conv = nn.ConvTranspose1d(out_depth,out_depth,1,bias=False)
        self.res_comb_norm = nn.BatchNorm1d(out_depth)
    def forward(self, input):
        output = F.leaky_relu(input)
        output = self.res_comb_norm(output)
        output = F.leaky_relu(output)
        output = self.res_cat_norm(output)
        output = self.sequence(output)
        return output


class Module18(nn.Module):
    #fewer step downs than module 3
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        n= 2
        self.encoder = nn.Sequential(
            Stack(3,n,3,6,1),
            Stack(5,n,6,12,2),
            Stack(11,n,12,18,2),
        )
        self.decoder = nn.Sequential(
            Stack_Transpose(11,n,12,18,2),
            Stack_Transpose(5,n,6,12,2),
            Stack_Transpose(3,n,3,6,1),
            nn.ConvTranspose1d(3,1,kernel_size=2,stride=2,padding=1),
        )
        self.to_flat = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(hp.dropout),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(19,16),
            nn.Dropout(hp.dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16,len(hp.classification_thresholds)+1)
        )
    @staticmethod
    def format_input(input):
        time_input = F.avg_pool1d(input['signal'],3,2,padding=1)
        min_len = min(time_input.shape[-1],input['fft'].shape[-1])
        x = torch.concat((time_input[...,:min_len],input['fft'][...,:min_len]),dim=1)
        return x
    def forward(self,input):
        x = self.format_input(input)
        x =  self.encoder(x)
        x = self.to_flat(x)
        x = torch.concatenate((input['num_layers'][...,None]/5,x),dim = -1)
        return self.linear(x)

class Module19(nn.Module):
    #fewer step downs than module 3
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        n= 2
        self.encoder = nn.Sequential(
            Stack(5,n,3,5,1),
            Stack(5,n,5,10,2),
            Stack(5,n,10,15,2),
            Stack(5,n,15,20,2),
            Stack(5,n,20,25,2),
        )
        self.decoder = nn.Sequential(
            Stack_Transpose(5,n,20,25,2),
            Stack_Transpose(5,n,15,20,2),
            Stack_Transpose(5,n,10,15,2),
            Stack_Transpose(5,n,5,10,2),
            Stack_Transpose(5,n,3,5,1),
            nn.ConvTranspose1d(3,1,kernel_size=2,stride=2,padding=1),
        )
        self.to_flat = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(hp.dropout),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(26,16),
            nn.Dropout(hp.dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16,len(hp.classification_thresholds)+1)
        )
    @staticmethod
    def format_input(input):
        time_input = F.avg_pool1d(input['signal'],3,2,padding=1)
        min_len = min(time_input.shape[-1],input['fft'].shape[-1])
        x = torch.concat((time_input[...,:min_len],input['fft'][...,:min_len]),dim=1)
        return x
    def forward(self,input):
        x = self.format_input(input)
        x =  self.encoder(x)
        x = self.to_flat(x)
        x = torch.concatenate((input['num_layers'][...,None]/5,x),dim = -1)
        return self.linear(x)