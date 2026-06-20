from dataclasses import dataclass, field
from typing import Callable, Literal
from torch.optim.sgd import SGD
from torch.nn import functional as F

import numpy as np
import torch.nn as nn
import torch
from typing import Any
from common import NUM_BLOCKS, NUM_LAYUPS


@dataclass
class Hyperparameters:
    name:str
    module:type[nn.Module]
    loss_function:Callable[...,torch.Tensor] = F.cross_entropy
    optimizer:'type[torch.optim.optimizer.Optimizer]' = SGD
    optimizer_config:dict[str,Any] = field(default_factory=dict)
    num_folds:int = 12
    butter_order:int = 6
    cutoff_frequency:float = 300*1000
    ply:Literal['3ply','5ply','poster','both'] = '3ply'
    ignore_proper_folds:bool = False
    batch_size:int = 32
    shuffle:bool = True
    num_epochs:int = 50
    nan_policy:Literal['remove'] = 'remove'
    out_type:Literal['edge','area'] = 'edge'
    out_shape:Literal['block','layer'] = 'block'
    classify:bool = True
    classification_thresholds:tuple[float,...] = (0.05,)
    out_boost_amp:float = 1
    "set to 0 for no boosting"
    data_balance_mode:Literal['balance','match'] = 'balance'
    "0-first,first-second,second-third,third-1, inclusive upwards"
    in_interpolation_mode:Literal['none'] = 'none'
    manual_class_weight:None|tuple[float,...] = None
    lr_scheduler:'type[torch.optim.lr_scheduler.LRScheduler]|None' = None
    lr_scheduler_config:dict[str,Any] = field(default_factory=dict)
    loss_config:dict[str,Any] = field(default_factory=dict)
    dropout:float = 0.1
    skip_incomplete_folds:bool = False
    early_exit_loss:None|float = None
    reorder_layups:None|tuple[int,int,int,int,int,int,int,int,int,int,int,int]|Literal['auto'] = None
    transfer_learning:bool|str = False
    transfer_learning_transform:Literal['waveform_in_painting','spec_rect_mask'] = 'waveform_in_painting'
    transfer_learning_boost:int = 5
    transfer_learning_transform_config:dict[str,Any] = field(default_factory=dict)
    transfer_epochs:int = 300
    use_parameters:tuple['str',...] = field(default_factory=tuple)
    spectrogram_config:dict[str,Any] = field(default_factory= lambda: dict(
        nperseg=120,
        noverlap=59,
        nfft=512,
        width=256,
        height=128
    ))
    freeze_encoder_epochs:int = 0
    append_poster_to_train:bool = False
    module_hyperparameters:dict = field(default_factory=dict)
    @property
    def num_layers(self) -> int:
        return int(self.ply[0])-1
    @property
    def allow_poster(self) -> bool:
        return self.ply == '3ply' and self.out_type == 'edge'
    @property
    def num_classes(self) -> int:
        return len(self.classification_thresholds)+1
    @property
    def num_blocks(self) -> int:
        if self.ply in ('3ply','5ply'):
            return NUM_BLOCKS
        elif self.ply == 'both':
            return NUM_BLOCKS*2
        else:
            raise NotImplementedError()
    @property
    def num_layups(self) -> int:
        if self.ply in ('3ply','5ply'):
            return NUM_LAYUPS
        elif self.ply == 'both':
            return NUM_LAYUPS*2
        else:
            raise NotImplementedError()
    @property
    def encoder_path(self) -> str:
        name = self.name
        if isinstance(self.transfer_learning,str):
            name = self.transfer_learning
        return f'temp/encoder_pretrain/model/{name}.pt'
    @property
    def encoder_loss_path(self) -> str:
        name = self.name
        if isinstance(self.transfer_learning,str):
            name = self.transfer_learning
        return f'temp/encoder_pretrain/train_loss/{name}.npy'
