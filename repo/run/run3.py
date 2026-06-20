import torch
import torch.utils.data
import torch_directml

from data.folds import get_folds
from run.data_log import Data_Log
from run.eval import Eval
from run.hyperparameters import Hyperparameters
from data.transfer import Transfer_D2
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
import warnings

def run3(hp:Hyperparameters):
    #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch_directml.device()
    with Data_Log(hp) as dl:
        if hp.transfer_learning and not os.path.exists(hp.encoder_path):
            transfer_loader  = torch.utils.data.DataLoader(
                    dataset=Transfer_D2(hp),
                    batch_size=hp.batch_size,
                    shuffle=hp.shuffle
                )
            model = hp.module(hp)
            transfer_model = model.transfer
            transfer_model.to(device)
            transfer_optimizer = hp.optimizer(
                transfer_model.parameters(),
                **hp.optimizer_config
            )
            total_loss = np.empty(hp.transfer_epochs)
            for e in tqdm(range(hp.transfer_epochs),leave=False):
                running_loss:torch.Tensor = torch.tensor(0.0, device=device)
                num_occurrences = 0
                for i,data in enumerate(transfer_loader):
                    if isinstance(data[0],dict):
                        inputs = {key:value.to(device) for key,value in data[0].items()}
                    else:
                        inputs = data[0].to(device)
                    labels = data[1].to(device)
                    num_occurrences += labels.shape[0]
                    outputs = transfer_model(inputs)
                    loss = F.l1_loss(outputs,labels)
                    running_loss += loss.detach()*labels.shape[0]
                    loss.backward()
                    transfer_optimizer.step()
                    transfer_optimizer.zero_grad()
                total_loss[e] = running_loss.item()/num_occurrences
            torch.save(model.encoder.state_dict(),hp.encoder_path)
            np.save(hp.encoder_loss_path,total_loss)
        eval_ = Eval(dl,device,hp.loss_function)
        for k,fold in enumerate(get_folds(hp)):
            if fold is None:
                continue
            (train_set,test_set) = fold
            dl.set_iter(k,0,False)
            #region: resetting model for new k-fold
            dl.log_train_freq(train_set)
            dl.log_test_freq(test_set)
                        #endregion
            #region: setting data into loaders
            train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=hp.batch_size,
                shuffle=hp.shuffle
                # sampler=torch.utils.data.WeightedRandomSampler(
                #     weights=train_weights,#type:ignore
                #     num_samples=len(train_weights)
                # )
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=hp.batch_size,
                shuffle=hp.shuffle
            )
            #endregion
            #region: training
            model = hp.module(hp)
            if hp.transfer_learning:
                with warnings.catch_warnings(action='ignore'):
                    model.encoder.load_state_dict(torch.load(hp.encoder_path))
            model.to(device)
            optimizer = hp.optimizer(
                model.parameters(),
                **hp.optimizer_config
            )
            scheduler = None
            if hp.lr_scheduler is not None:
                scheduler = hp.lr_scheduler(optimizer,**hp.lr_scheduler_config)
            eval_.set_fold(model,optimizer)
            if hp.freeze_encoder_epochs > 0:
                model.freeze_encoder()
            for e in range(hp.num_epochs+1):
                if e == hp.freeze_encoder_epochs:
                    model.unfreeze_encoder()
                dl.set_iter(k,e)
                #skip training on first epoch for untrained metrics
                early_exit = eval_(train_loader,'train',train=e!=0)
                eval_(test_loader,'test')
                if scheduler is not None and e != 0:
                    scheduler.step()
                if early_exit:
                    break
            #endregion

