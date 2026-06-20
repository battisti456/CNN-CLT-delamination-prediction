import torch.nn as nn
import torch
from torch.nn import _reduction as _Reduction, functional as F
from run.hyperparameters import Hyperparameters
from typing import Literal, Self


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.relu2 = nn.ReLU()

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) \
            if n_inputs != n_outputs else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        # crop to seq length
        return out[..., :x.size(-1)] + residual


class TCNEncoder(nn.Module):
    def __init__(self, channels, num_layers=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(TemporalBlock(
                channels, channels, kernel_size=3, dilation=dilation
            ))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x = (batch, seq, channels)
        x = x.transpose(1, 2)      # → (batch, channels, seq)
        x = self.net(x)
        return x.transpose(1, 2)   # back → (batch, seq, channels)



class Qu_CNN_BiGRU_Feature_Extraction_CNN(nn.Sequential):
    def __init__(self,hp:Hyperparameters,domain:Literal['time','freq']):
        super().__init__()#significantly fewer sampling points than our case, maybe optimizations here?
        self.hp = hp
        #7928
        self.conv1 = nn.Conv1d(1 if domain == 'time' else 2,16,64,stride=8)
        #984, supposed to be 121
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(16,16,3,stride = 2)
        #491, supposed to be 60
        self.relu2 = nn.ReLU()#paper does not specify where activation functions are placed, specifically
        self.pool = nn.MaxPool1d(2,stride = 2)
        #245, supposed to be 30

class Qu_CNN_BiGRU_Feature_Extraction(nn.Module):
    def __init__(self,hp:Hyperparameters,domain:Literal['time','freq']):
        super().__init__()
        self.hp = hp
        self.cnn = Qu_CNN_BiGRU_Feature_Extraction_CNN(hp,domain)
        shape = 16
        self.attn = nn.MultiheadAttention(
            embed_dim=shape,
            num_heads=4,#does not seem specified,
            batch_first=True,
        )
        # self.bigru = nn.GRU(
        #     input_size=shape,#?????
        #     hidden_size=shape,
        #     num_layers=2,
        #     bidirectional=True,
        #     batch_first=True,
        # ).to('cpu',torch.float32)
        # self.bigru = nn.GRU(
        #     input_size=shape,#?????
        #     hidden_size=shape,
        #     num_layers=2,
        #     bidirectional=True,
        #     batch_first=True
        # ).to('cpu',torch.float32)
        self.bigru = TCNEncoder(channels=shape)

    # def _apply(self, fn):#type:ignore
    #     super()._apply(fn)
    #     self.bigru.to('cpu',torch.float32)

    #     for p in self.bigru.parameters():
    #         if p.grad is not None:
    #             p.grad = p.grad.to(device='cpu', dtype=torch.float32)

    #     return self

    def forward(self,input:torch.Tensor):
        output = self.cnn(input)
        output = output.permute(0, 2, 1).contiguous()
        output,_ = self.attn(output,output,output,need_weights=False)#self attention
        # device,dtype = output.device, output.dtype
        # output,_ = self.bigru(output.to('cpu',torch.float32))#type:ignore
        # output = output.to(device,dtype)
        output = self.bigru(output)
        return output

class Qu_CNN_BiGRU_Head(nn.Module):
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        self.hp = hp
        self.fe_time = Qu_CNN_BiGRU_Feature_Extraction(hp,'time')
        self.fe_freq = Qu_CNN_BiGRU_Feature_Extraction(hp,'freq')
    def extract_inputs(self,input:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        time_input = F.avg_pool1d(input,3,2,padding=1)
        freq_input = torch.view_as_real(torch.fft.rfft(input.to('cpu'),dim = -1))
        freq_input = freq_input.permute(0, 3, 1, 2).squeeze(2).contiguous().to(input.device)
        return time_input,freq_input
    def forward(self,input:torch.Tensor) -> torch.Tensor:
        time_input,freq_input = self.extract_inputs(input)
        fe_time_out = self.fe_time(time_input)
        fe_freq_out = self.fe_freq(freq_input)
        output = torch.concatenate((fe_time_out,fe_freq_out),dim = 1)
        return output

class Qu_CNN_BiGRU_SVM(nn.Sequential):
    "not actually svm, meant to act as an alternative since mixing deep learning and SVMs is difficult"
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        self.hp = hp
        self.linear = nn.Linear(242,hp.num_classes)

class Qu_CNN_BiGRU(nn.Module):
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        self.hp = hp
        self.head = Qu_CNN_BiGRU_Head(hp)
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.svm = Qu_CNN_BiGRU_SVM(hp)
    def forward(self,input):
        output = self.head(input)
        output = self.gap(output)
        output = self.svm(output.squeeze(-1))
        return output
Module9 = Qu_CNN_BiGRU