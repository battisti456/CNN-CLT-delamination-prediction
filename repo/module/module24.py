import torch.nn as nn
import torch
from torch.nn import _reduction as _Reduction, functional as F
from run.hyperparameters import Hyperparameters

from common import SIGNAL_LENGTH


class Module24(nn.Module):
    #fewer step downs than module 3
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear((SIGNAL_LENGTH//2)*3+1,32),
            nn.Dropout(hp.dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32,len(hp.classification_thresholds)+1)
        )
    def forward(self,input):
        time_input = F.avg_pool1d(input['signal'],3,2,padding=1)
        min_len = min(time_input.shape[-1],input['fft'].shape[-1])
        x = torch.concat((time_input[...,:min_len],input['fft'][...,:min_len]),dim=1)
        x = torch.concat((x.flatten(1),input['num_layers'][:,None]),dim=-1)
        return self.linear(x)