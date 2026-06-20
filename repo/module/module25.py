import torch.nn as nn
import torch
from torch.nn import _reduction as _Reduction, functional as F
from run.hyperparameters import Hyperparameters

class Stack_Sequence(nn.Sequential):
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

class Stack(nn.Module):
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
        self.sequence = Stack_Sequence(k,num_layers,in_depth,out_depth,stride)
        self.gate = nn.Conv1d(in_depth,out_depth,1,stride=stride,bias=False)
        self.res_cat_norm = nn.BatchNorm1d(out_depth)
        self.res_comb_conv = nn.Conv1d(out_depth,out_depth,1,bias=False)
        #self.res_comb_norm = nn.GroupNorm(num_groups=min(5,self._out_depth),num_channels=out_depth)
        self.res_comb_norm = nn.BatchNorm1d(out_depth)
    def forward(self, input):
        output = self.sequence(input)
        res = self.gate(input)
        if output.size(-1) > res.size(-1):
            output = output[...,:res.size(-1)]
        elif res.size(-1) > output.size(-1):
            res = res[...,:output.size(-1)]
        g = torch.sigmoid(res)
        output =  g*output+(1-g)*res
        output = self.res_cat_norm(output)
        output = F.leaky_relu(output)
        output = self.res_comb_conv(output)
        output = self.res_comb_norm(output)
        output = F.leaky_relu(output)
        return output

class Module25(nn.Module):
    #fewer step downs than module 3
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        n= 2
        self.sequence = nn.Sequential(
            Stack(5,n,1,5,1),
            Stack(5,n,5,10,2),
            Stack(5,n,10,15,2),
            Stack(5,n,15,20,2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(20,16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16,len(hp.classification_thresholds)+1)
        )
    def forward(self,input):
        return self.sequence(input['signal'])