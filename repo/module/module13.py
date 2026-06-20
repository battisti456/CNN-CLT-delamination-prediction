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

class Module13(nn.Module):
    #fewer step downs than module 3
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        n= 2
        self.sequence = nn.Sequential(
            Stack(5,n,3,5,1),
            Stack(5,n,5,10,2),
            Stack(5,n,10,15,2),
            Stack(5,n,15,20,2),
            Stack(5,n,20,25,2),
            Stack(5,n,25,30,2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(30,16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16,len(hp.classification_thresholds)+1)
        )
    def forward(self,input):
        time_input = F.avg_pool1d(input,3,2,padding=1)
        freq_input = torch.view_as_real(torch.fft.rfft(input.to('cpu'),dim = -1))
        freq_input = freq_input.permute(0, 3, 1, 2).squeeze(2).contiguous()
        input = torch.concat((time_input.cpu(),freq_input[...,:-1]),dim=1).to(input.device)
        return self.sequence(input)
    
class Module14(nn.Module):
    #fewer step downs than module 3
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        n= 2
        self.sequence = nn.Sequential(
            Stack(5,n,3,5,1),
            Stack(5,n,5,10,2),
            Stack(5,n,10,15,2),
            Stack(5,n,15,20,2),
            Stack(5,n,20,25,2),
            Stack(5,n,25,30,2),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(hp.dropout),
            nn.Flatten(),
            nn.Linear(30,16),
            nn.Dropout(hp.dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16,len(hp.classification_thresholds)+1)
        )
    def forward(self,input):
        time_input = F.avg_pool1d(input['signal'],3,2,padding=1)
        min_len = min(time_input.shape[-1],input['fft'].shape[-1])
        x = torch.concat((time_input[...,:min_len],input['fft'][...,:min_len]),dim=1)
        v =  self.sequence(x)
        return v

class Module15(nn.Module):
    #fewer step downs than module 3
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        n= 2
        self.cnn_stacks = nn.Sequential(
            Stack(5,n,3,5,1),
            Stack(5,n,5,10,2),
            Stack(5,n,10,15,2),
            Stack(5,n,15,20,2),
            Stack(5,n,20,25,2),
            Stack(5,n,25,30,2),
        )
        self.to_flat = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(hp.dropout),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(31,16),
            nn.Dropout(hp.dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16,len(hp.classification_thresholds)+1)
        )
    def forward(self,input):
        time_input = F.avg_pool1d(input['signal'],3,2,padding=1)
        min_len = min(time_input.shape[-1],input['fft'].shape[-1])
        x = torch.concat((time_input[...,:min_len],input['fft'][...,:min_len]),dim=1)
        x =  self.cnn_stacks(x)
        x = self.to_flat(x)
        x = torch.concatenate((input['num_layers'][...,None]/5,x),dim = -1)
        return self.linear(x)

class Module16(nn.Module):
    #fewer step downs than module 3
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        n= 2
        self.cnn_stacks = nn.Sequential(
            Stack(5,n,3,8,1),
            Stack(5,n,8,16,2),
            Stack(5,n,16,32,2),
        )
        self.to_flat = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(hp.dropout),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(33,16),
            nn.Dropout(hp.dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16,len(hp.classification_thresholds)+1)
        )
    def forward(self,input):
        time_input = F.avg_pool1d(input['signal'],3,2,padding=1)
        min_len = min(time_input.shape[-1],input['fft'].shape[-1])
        x = torch.concat((time_input[...,:min_len],input['fft'][...,:min_len]),dim=1)
        x =  self.cnn_stacks(x)
        x = self.to_flat(x)
        x = torch.concatenate((input['num_layers'][...,None]/5,x),dim = -1)
        return self.linear(x)

class Module17(nn.Module):
    #fewer step downs than module 3
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        n= 2
        self.cnn_stacks = nn.Sequential(
            Stack(3,n,3,6,1),
            Stack(5,n,6,12,2),
            Stack(11,n,12,18,2),
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
    def forward(self,input):
        time_input = F.avg_pool1d(input['signal'],3,2,padding=1)
        min_len = min(time_input.shape[-1],input['fft'].shape[-1])
        x = torch.concat((time_input[...,:min_len],input['fft'][...,:min_len]),dim=1)
        x =  self.cnn_stacks(x)
        x = self.to_flat(x)
        x = torch.concatenate((input['num_layers'][...,None]/5,x),dim = -1)
        return self.linear(x)