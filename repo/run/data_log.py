import numpy as np
from run.hyperparameters import Hyperparameters
import pickle
from tqdm import tqdm
from typing import Self
from data.folds import D2
from data.misc import class_freq

class Data_Log:
    def __init__(
            self,
            hp:Hyperparameters
    ):
        self.hp = hp
        self.metrics:dict[str,np.ndarray[tuple[int,int],np.dtype[np.float32]]] = {}
        self.train_class_freq = np.empty((hp.num_folds,hp.num_classes),np.intp)
        "-1 indicates unset"
        self.test_class_freq = np.empty((hp.num_folds,hp.num_classes),np.intp)
        "-1 indicates unset"
        self.test_class_freq[:] = -1#initialize to -1 to indicate unset
        self.train_class_freq[:] = -1
        self._fold:int = 0
        self._epoch:int = 0
        self._last:int = 0
    def current_train_freq(self)-> np.ndarray[tuple[int],np.dtype[np.intp]]:
        return self.train_class_freq[self._fold,:]
    def current_test_freq(self)-> np.ndarray[tuple[int],np.dtype[np.intp]]:
        return self.test_class_freq[self._fold,:]
    def log_metric(self,name:str,metric:float):
        if name not in self.metrics:
            self.metrics[name] = np.empty((self.hp.num_folds,self.hp.num_epochs+1),np.float32)
            self.metrics[name][:] = np.nan
        self.metrics[name][self._fold, self._epoch] = metric
    def log_metrics(self,metrics:dict[str,float]):
        for key,value in metrics.items():
            self.log_metric(key,value)
    def log_train_freq(self,dataset:D2) -> np.ndarray[tuple[int],np.dtype[np.intp]]:
        return self._log_freq(dataset,self.train_class_freq)
    def log_test_freq(self,dataset:D2) -> np.ndarray[tuple[int],np.dtype[np.intp]]:
        return self._log_freq(dataset,self.test_class_freq)
    def _log_freq(
            self,
            dataset:D2,
            dest:np.ndarray[tuple[int,int],np.dtype[np.intp]]
        ) -> np.ndarray[tuple[int],np.dtype[np.intp]]:
        dest[self._fold,:] = class_freq(np.array(dataset.labels),self.hp.num_classes)#type:ignore
        return dest[self._fold,:]
    def set_iter(self,k:int,e:int,update:bool = True):
        if hasattr(self,'bar') and update:
            current = k*self.hp.num_epochs+e
            self.bar.update(current-self._last)
            self._last = current
        self._fold = k
        self._epoch = e
    def save(self):
        with open(f'data_logs/{self.hp.name}.pkl','wb') as file:
            pickle.dump(self,file,pickle.HIGHEST_PROTOCOL)
    def __enter__(self) -> Self:
        self.bar = tqdm(total=self.hp.num_folds*(self.hp.num_epochs+1))
        self.bar.__enter__()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.bar.__exit__(exc_type, exc, tb)
        del self.bar
        self.save()
    @classmethod
    def load(cls,name:str) -> 'Data_Log':
        with open(f'data_logs/{name}.pkl', 'rb') as file:
            return pickle.load(file)