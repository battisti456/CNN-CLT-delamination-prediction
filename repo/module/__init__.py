import torch
from run.hyperparameters import Hyperparameters
from typing import Protocol, TypeVar, Iterable

hp_var = TypeVar('hp_var',bound = Hyperparameters, covariant=True)

class HPModule(Protocol[hp_var]):
    _hp:hp_var
    def __init__(self,hp:hp_var):
        ...
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        ...
    def to(self,device:'torch.Device'):
        ...
    def __call__(self,x:torch.Tensor) -> torch.Tensor:
        ...
    def parameters(self) -> Iterable[torch.Tensor]:
        ...