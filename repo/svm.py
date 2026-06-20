from data.folds_svc import get_folds
import sklearn.svm
from tqdm import tqdm
import numpy as np
import torchmetrics as tm
import torch
from run.hyperparameters import Hyperparameters
import pickle

class SVC_Data_Log:
    def __init__(self,hp:Hyperparameters,model = 'svc'):
        self.hp = hp
        self.metrics:dict[str,np.ndarray[tuple[int],np.dtype[np.float32]]] = {}
        self._fold = 0
        self.model = model
    def set_iter(self,k:int):
        self._fold = k
    def log_metric(self,name:str,metric:float):
        if name not in self.metrics:
            self.metrics[name] = np.empty(self.hp.num_folds,np.float32)
            self.metrics[name][:] = np.nan
        self.metrics[name][self._fold] = metric
    def log_metrics(self,metrics:dict[str,float]):
        for key,value in metrics.items():
            self.log_metric(key,value)
    def save(self):
        with open(f'data_logs/{self.model}/{self.hp.name}.pkl','wb') as file:
            pickle.dump(self,file,pickle.HIGHEST_PROTOCOL)
    @classmethod
    def load(cls,name:str,model:str = 'svc') -> 'SVC_Data_Log':
        with open(f'data_logs/{model}/{name}.pkl', 'rb') as file:
            return pickle.load(file)

if __name__ == '__main__':
    from main import hp
    model = sklearn.svm.SVC(probability=True)
    dl = SVC_Data_Log(hp)
    metrics = tm.MetricCollection(dict(
        auroc = tm.AUROC(task='multiclass',num_classes=hp.num_classes),
        f1 = tm.F1Score(task='multiclass',num_classes=hp.num_classes),
        acc = tm.Accuracy(task='multiclass',num_classes=hp.num_classes),
        _acc = tm.Accuracy(task='multiclass',num_classes=hp.num_classes, average = 'none'),
        _prec = tm.Precision(task='multiclass',num_classes=hp.num_classes, average = 'none'),
        _recall =tm.Recall(task='multiclass',num_classes=hp.num_classes, average = 'none'),
        prec = tm.Precision(task='multiclass',num_classes=hp.num_classes),
        recall =tm.Recall(task='multiclass',num_classes=hp.num_classes),
    ))
    for k, val in tqdm(enumerate(get_folds(hp)),total=hp.num_folds):
        dl.set_iter(k)
        if val is None:
            continue
        train_data, test_data = val
        min_rmsv = np.min(train_data.rmsv)
        max_rmsv = np.max(train_data.rmsv)
        train_rmsv = (train_data.rmsv-min_rmsv)/(max_rmsv-min_rmsv)
        test_rmsv = (test_data.rmsv-min_rmsv)/(max_rmsv-min_rmsv)
        min_mcf = np.min(train_data.mcfs)
        max_mcf = np.max(train_data.mcfs)
        train_mcf = (train_data.mcfs-min_mcf)/(max_mcf-min_mcf)
        test_mcf = (test_data.mcfs-min_mcf)/(max_mcf-min_mcf)
        train_data_vec = np.concatenate((train_rmsv,train_mcf,train_data.num_layers[:,None] == 4),axis = -1)
        test_data_vec = np.concatenate((test_rmsv,test_mcf,test_data.num_layers[:,None] == 4),axis = -1)
        model.fit(train_data_vec,train_data.labels)
        pred_train = model.predict_proba(train_data_vec)
        pred_test = model.predict_proba(test_data_vec)
        metrics.update(torch.tensor(pred_train),torch.tensor(train_data.labels))
        train_metrics = metrics.compute()
        metrics.reset()
        metrics.update(torch.tensor(pred_test),torch.tensor(test_data.labels))
        test_metrics = metrics.compute()
        metrics.reset()
        for name, met in (
            ('train', train_metrics),
            ('test', test_metrics)
        ):
            for prop, val in met.items():
                assert isinstance(val,torch.Tensor)
                if val.ndim == 0:
                    dl.log_metric(f"{name}_{prop}",val.item())
                    continue
                for i in range(len(val)):
                    dl.log_metric(f"{name}_class{i}{prop}",val[i].item())
    dl.save()



