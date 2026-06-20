from contextlib import nullcontext
from typing import TypeVar, overload

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import _reduction as _Reduction
from torch.nn import functional as F
from tqdm import tqdm

from data.folds import get_folds
from module import HPModule
from run.hyperparameters import Hyperparameters

hp_var = TypeVar('hp_var',bound = Hyperparameters, covariant=True)

class Store[T]:
    def __init__(self,name):
        self.values:list[list[T]] = []
        self.name = name
    def __len__(self) -> int:
        return len(self.values)
    @overload
    def __getitem__(self, index:int) -> list[T]: ...
    @overload
    def __getitem__(self, index:tuple[int,int]) -> T: ...
    def __getitem__(self, index:int|tuple[int,int]) -> list[T]|T:
        if isinstance(index,int):
            return self.values[index]
        else:
            return self.values[index[0]][index[1]]
    def __setitem__(self, index:tuple[int,int], value):
        while len(self.values) < index[0]+1:
            self.values.append([])
        if len(self.values[index[0]]) < index[1] + 1:
            self.values[index[0]].append(value)
        else:
            self.values[index[0]][index[1]] = value
    def __iter__(self):
        return self.values.__iter__()
    def mean(self) -> np.ndarray:
        return np.mean(np.array(self.values),axis = 0)

def run1(module:type[HPModule[hp_var]],hp:hp_var):
    model:torch.nn.Module
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    optimizer:torch.optim.SGD
    loss_function = F.cross_entropy
    class_weights:torch.Tensor
    def _eval(
            data_loader:torch.utils.data.DataLoader,
            loss_store:Store[float],
            acc_store:Store[float],
            k:int,
            e:int,
            train:bool = False
    ):
        with nullcontext() if train else torch.no_grad():
            running_loss = 0
            running_correct = 0
            for i,data in enumerate(data_loader):
                inputs, labels = data[:][0].to(device), data[:][1].to(device)
                outputs = model.forward(inputs)
                loss = loss_function(outputs,labels[:,0],weight=class_weights)
                running_loss += loss.item()
                running_correct += (outputs.argmax(dim=1) == labels[:,0]).sum().item()
                if train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            loss_store[k,e] = running_loss/len(data_loader.dataset)#type:ignore
            acc_store[k,e] = running_correct/len(data_loader.dataset)#type:ignore
    train_loss:Store[float] = Store("Train Loss")
    train_acc:Store[float] = Store("Train Acc")
    test_loss:Store[float] = Store("Test Loss")
    test_acc:Store[float] = Store("Test Acc")
    if hp.allow_poster:
        poster_loss:Store[float] = Store("Poster Loss")
        poster_acc:Store[float] = Store("Poster Acc")
    with tqdm(total=hp.num_folds*hp.num_epochs) as bar:
        for k,(train_set,test_set,poster_set) in enumerate(get_folds(hp)):
            model = module(hp)
            optimizer = torch.optim.SGD(#which optimizer?
                model.parameters(),
                momentum=hp.momentum,
                weight_decay=hp.decay,
            )
            train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=hp.batch_size,
                shuffle=hp.shuffle
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=hp.batch_size,
                shuffle=hp.shuffle
            )
            if hp.allow_poster:
                assert poster_set is not None
                poster_loader = torch.utils.data.DataLoader(
                    dataset=poster_set,
                    batch_size=hp.batch_size,
                    shuffle=hp.shuffle
                )
            class_freqs = torch.zeros(hp.num_classes)
            for i,data in enumerate(train_loader):
                for j in range(hp.num_classes):
                    class_freqs[j] += (data[:][1] == j).sum()
            # class_weights = 1/(class_freqs/class_freqs.sum())**0.5
            # class_weights = torch.tensor([class_freqs[0]/class_freqs[1],1])
            _eval(train_loader,train_loss,train_acc,k,0)#establish random start value
            _eval(test_loader,test_loss,test_acc,k,0)#establish random start value
            if hp.allow_poster:
                _eval(poster_loader,poster_loss,poster_acc,k,0)#type:ignore
            for e in range(hp.num_epochs):
                _eval(train_loader,train_loss,train_acc,k,e+1,train=True)
                _eval(test_loader,test_loss,test_acc,k,e+1)
                if hp.allow_poster:
                    _eval(poster_loader,poster_loss,poster_acc,k,e+1)#type:ignore
                bar.update(1)
    import matplotlib.pyplot as plt

    class_freqs = torch.zeros(hp.num_classes)
    for i,data in enumerate(train_loader):#type:ignore
        for j in range(hp.num_classes):
            class_freqs[j] += (data[:][1] == j).sum()
    for i,data in enumerate(test_loader):#type:ignore
        for j in range(hp.num_classes):
            class_freqs[j] += (data[:][1] == j).sum()
    class_acc = class_freqs/class_freqs.sum()
    
    for store in (
        train_loss,
        train_acc,
        test_loss,
        test_acc,
    ) + (() if not hp.allow_poster else (
        poster_loss,#type:ignore
        poster_acc#type:ignore
    )):
        fig = plt.figure()
        ax = fig.subplots()
        # for fold in store:
        #     ax.plot(fold,zorder = 0)
        ax.plot(store.mean())
        ax.set_xlabel('Epochs')
        ax.set_ylabel(store.name)
        if 'Acc' in store.name:
            ax.hlines(
                class_acc[0],
                xmin=0,
                xmax=hp.num_epochs,
                label='ACC if always guess pass',
                zorder = -1,
            )
            ax.legend()
    plt.show()

