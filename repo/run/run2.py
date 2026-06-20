import torch
import torch.utils.data
import torch_directml

from data.folds import get_folds
from run.data_log import Data_Log
from run.eval import Eval
from run.hyperparameters import Hyperparameters

def run2(hp:Hyperparameters):
    #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch_directml.device()
    with Data_Log(hp) as dl:
        eval_ = Eval(dl,device,hp.loss_function)
        for k,fold in enumerate(get_folds(hp)):
            if fold is None:
                continue
            (train_set,test_set) = fold
            dl.set_iter(k,0,False)
            #region: resetting model for new k-fold
            dl.log_train_freq(train_set)
            dl.log_test_freq(test_set)
            model = hp.module(hp)
            model.to(device)
            optimizer = hp.optimizer(
                model.parameters(),
                **hp.optimizer_config
            )
            scheduler = None
            if hp.lr_scheduler is not None:
                scheduler = hp.lr_scheduler(optimizer,**hp.lr_scheduler_config)
            eval_.set_fold(model,optimizer)
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
            for e in range(hp.num_epochs+1):
                dl.set_iter(k,e)
                #skip training on first epoch for untrained metrics
                early_exit = eval_(train_loader,'train',train=e!=0)
                eval_(test_loader,'test')
                if scheduler is not None and e != 0:
                    scheduler.step()
                if early_exit:
                    break
            #endregion

