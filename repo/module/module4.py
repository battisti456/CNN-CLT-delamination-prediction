import torch.nn as nn
import torch
from torch.nn import _reduction as _Reduction, functional as F
from run.hyperparameters import Hyperparameters

class Module4_Stack_Sequence(nn.Sequential):
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
                self.append(
                    nn.Conv1d(in_depth if i==0 else out_depth,out_depth,k,stride=stride if i == 0 else 1, padding=2,bias=False)
                )
                #self.append(nn.GroupNorm(num_groups=min(5,out_depth),num_channels=out_depth))
                self.append(nn.BatchNorm1d(out_depth))
                self.append(nn.LeakyReLU(inplace=True))

class Module4_Stack(nn.Module):
    def __init__(
            self,
            k:int,
            num_layers:int,
            in_depth:int,
            out_depth:int,
            stride:int
    ):
        self._k = k
        self._in_depth = in_depth
        self._out_depth = out_depth
        self._res_depth = self._in_depth+self._out_depth
        self._stride = stride
        super().__init__()
        self.sequence = Module4_Stack_Sequence(k,num_layers,in_depth,out_depth,stride)
        if self._stride > 1:
            self.res_down_conv = nn.Conv1d(in_depth,in_depth,1,stride=stride,bias=False)
        self.res_cat_norm = nn.BatchNorm1d(out_depth+in_depth)
        self.res_comb_conv = nn.Conv1d(in_depth+out_depth,out_depth,1,bias=False)
        #self.res_comb_norm = nn.GroupNorm(num_groups=min(5,self._out_depth),num_channels=out_depth)
        self.res_comb_norm = nn.BatchNorm1d(out_depth)
    def forward(self, input):
        output = self.sequence.forward(input)
        res = input
        if self._stride > 1:
            res = self.res_down_conv(res)
        output =  torch.cat((output,res),dim = 1)
        output = self.res_cat_norm.forward(output)
        output = F.leaky_relu(output)
        output = self.res_comb_conv.forward(output)
        output = self.res_comb_norm.forward(output)
        output = F.leaky_relu(output)
        return output

class Module4(nn.Sequential):
    #fewer step downs than module 3
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        n= 2
        _layers = [#ai recommends faster downsampling
            Module4_Stack(5,n,1,5,1),
            Module4_Stack(5,n,5,10,4),
            Module4_Stack(5,n,10,15,4),
            Module4_Stack(5,n,15,20,4),
            Module4_Stack(5,n,20,25,4),
            Module4_Stack(5,n,25,30,4),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(30,16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16,len(hp.classification_thresholds)+1)
        ]
        for _layer in _layers:
            self.append(_layer)
class Module5(nn.Sequential):
    #fewer step downs than module 3
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        n= 2
        _layers = [#ai recommends faster downsampling
            Module4_Stack(5,n,1,5,1),
            Module4_Stack(5,n,5,10,6),
            Module4_Stack(5,n,10,15,6),
            Module4_Stack(5,n,15,20,6),
            Module4_Stack(5,n,20,25,6),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(25,16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16,len(hp.classification_thresholds)+1)
        ]
        for _layer in _layers:
            self.append(_layer)
class Module6(nn.Sequential):
    #fewer step downs than module 3
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        n= 2
        _layers = [#ai recommends faster downsampling
            Module4_Stack(5,n,1,5,6),
            Module4_Stack(5,n,5,10,6),
            Module4_Stack(5,n,10,15,6),
            Module4_Stack(5,n,15,20,6),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(20,16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16,len(hp.classification_thresholds)+1)
        ]
        for _layer in _layers:
            self.append(_layer)
class Module7(nn.Sequential):
    #fewer step downs than module 3
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        n= 2
        _layers = [#ai recommends faster downsampling
            Module4_Stack(5,n,1,5,10),
            Module4_Stack(5,n,5,10,10),
            Module4_Stack(5,n,10,15,10),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(15,16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16,len(hp.classification_thresholds)+1)
        ]
        for _layer in _layers:
            self.append(_layer)