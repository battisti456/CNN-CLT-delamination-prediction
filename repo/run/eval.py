from contextlib import nullcontext
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from run.data_log import Data_Log
import torchmetrics as tm

class Eval:
    def __init__(self,dl:Data_Log,device:str,loss_function:Callable[..., torch.Tensor]):
        self.dl = dl
        self.device = device
        self.loss_function:Callable[..., torch.Tensor] = loss_function
        self.metric_logs:dict[str,list[list[float]]] = {}

        self.model:nn.Module
        self.optimizer:'torch.optim.optimizer.Optimizer'
    def _log_metric(self,name,value):
        if name not in self.metric_logs:
            self.metric_logs[name] = [[]]
        self.metric_logs[name][-1].append(value)
    def set_fold(self,model:nn.Module,optimizer:'torch.optim.optimizer.Optimizer'):
        self.model = model
        self.optimizer = optimizer
        for val in self.metric_logs.values():
            val.append([])
    def __call__(
            self,
            data_loader:torch.utils.data.DataLoader,
            name:str,
            train:bool = False
        ) -> bool:
            with nullcontext() if train else torch.no_grad():
                running_loss:torch.Tensor = torch.tensor(0.0, device=self.device)
                num_occurrences = 0
                metrics = tm.MetricCollection(dict(
                    auroc = tm.AUROC(task='multiclass',num_classes=self.dl.hp.num_classes),
                    f1 = tm.F1Score(task='multiclass',num_classes=self.dl.hp.num_classes),
                    acc = tm.Accuracy(task='multiclass',num_classes=self.dl.hp.num_classes),
                    _acc = tm.Accuracy(task='multiclass',num_classes=self.dl.hp.num_classes, average = 'none'),
                    _prec = tm.Precision(task='multiclass',num_classes=self.dl.hp.num_classes, average = 'none'),
                    _recall =tm.Recall(task='multiclass',num_classes=self.dl.hp.num_classes, average = 'none'),
                    prec = tm.Precision(task='multiclass',num_classes=self.dl.hp.num_classes),
                    recall =tm.Recall(task='multiclass',num_classes=self.dl.hp.num_classes),
                ))
                if train:
                    self.model.train()
                else:
                    self.model.eval()
                for i,data in enumerate(data_loader):
                    inputs = {key:value.to(self.device) for key,value in data[0].items()}
                    labels = data[1].to(self.device)
                    outputs = self.model(inputs)
                    loss = self.loss_function(
                        outputs,
                        labels,
                        **self.dl.hp.loss_config | (dict(
                            weight=torch.tensor(
                                self.dl.hp.manual_class_weight,
                                device=self.device
                            )
                        )if self.dl.hp.manual_class_weight is not None else {})
                    )
                    metrics.update(outputs.cpu(),labels.cpu())
                    running_loss += loss.detach()*labels.shape[0]
                    num_occurrences += labels.shape[0]
                    if train:
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                for prop, val in metrics.compute().items():
                    assert isinstance(val,torch.Tensor)
                    if val.ndim == 0:
                        self.dl.log_metric(f"{name}_{prop}",val.item())
                        continue
                    for i in range(len(val)):
                        self.dl.log_metric(f"{name}_class{i}{prop}",val[i].item())
                mean_loss = (running_loss/num_occurrences).item()
                self.dl.log_metric(f"{name}_loss",mean_loss)
            if self.dl.hp.early_exit_loss is not None:
                if mean_loss < self.dl.hp.early_exit_loss:
                    return True
            return False