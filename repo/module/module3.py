import torch.nn as nn
import torch
from torch.nn import _reduction as _Reduction, functional as F
from run.hyperparameters import Hyperparameters

class Module3_Stack_Sequence(nn.Sequential):
    def __init__(
                self,
                k:int,
                num_layers:int,
                in_depth:int,
                out_depth:int,
                downsample:bool = False,
        ):
            super().__init__()
            for i in range(num_layers):
                self.append(
                    nn.Conv1d(in_depth if i==0 else out_depth,out_depth,k,stride=2 if downsample and i ==0 else 1, padding=2,bias=False)
                )
                #self.append(nn.GroupNorm(num_groups=min(5,out_depth),num_channels=out_depth))
                self.append(nn.BatchNorm1d(out_depth))
                self.append(nn.LeakyReLU(inplace=True))

class Module3_Stack(nn.Module):
    def __init__(
            self,
            k:int,
            num_layers:int,
            in_depth:int,
            out_depth:int,
            downsample:bool = False
    ):
        self._k = k
        self._in_depth = in_depth
        self._out_depth = out_depth
        self._res_depth = self._in_depth+self._out_depth
        self._downsample = downsample
        super().__init__()
        self.sequence = Module3_Stack_Sequence(k,num_layers,in_depth,out_depth,downsample)
        if self._downsample:
            self.res_down_conv = nn.Conv1d(in_depth,in_depth,1,stride=2,bias=False)
        self.res_cat_norm = nn.BatchNorm1d(out_depth+in_depth)
        self.res_comb_conv = nn.Conv1d(in_depth+out_depth,out_depth,1,bias=False)
        #self.res_comb_norm = nn.GroupNorm(num_groups=min(5,self._out_depth),num_channels=out_depth)
        self.res_comb_norm = nn.BatchNorm1d(out_depth)
    def forward(self, input):
        output = self.sequence.forward(input)
        res = input
        if self._downsample:
            res = self.res_down_conv(res)
        output =  torch.cat((output,res),dim = 1)
        output = self.res_cat_norm.forward(output)
        output = F.leaky_relu(output)
        output = self.res_comb_conv.forward(output)
        output = self.res_comb_norm.forward(output)
        output = F.leaky_relu(output)
        return output

class Module3(nn.Sequential):
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        n= 2
        _layers = [#ai recommends faster downsampling
            #7928
            Module3_Stack(5,n,1,5,False),
            #7928
            Module3_Stack(5,n,5,10,True),
            #3964
            Module3_Stack(5,n,10,15,True),
            #1982
            Module3_Stack(5,n,15,20,True),
            #991
            Module3_Stack(5,n,20,25,True),
            #496
            Module3_Stack(5,n,25,30,True),
            #248
            Module3_Stack(5,n,30,35,True),
            #124
            Module3_Stack(5,n,35,40,True),
            #62
            Module3_Stack(5,n,40,45,True),
            #31
            Module3_Stack(5,n,45,50,True),
            #16
            Module3_Stack(5,n,50,55,True),
            #8
            Module3_Stack(5,n,55,60,True),
            #4
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(60,32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32,32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32,len(hp.classification_thresholds)+1)
        ]
        for _layer in _layers:
            self.append(_layer)