import torch.nn as nn

from run.hyperparameters import Hyperparameters


class Module2(nn.Sequential):
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        _layers = [
            #7928
            nn.ZeroPad1d(6),
            #7940
            nn.Conv1d(1,5,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(5,5,5,padding = 2),
            #nn.LeakyReLU(),
            nn.Conv1d(5,5,5,padding = 2),
            nn.Conv1d(5,5,5,stride=2,padding = 7),
            #3980
            nn.Conv1d(5,10,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(10,10,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(10,10,5,padding=2),
            nn.Conv1d(10,10,5,stride = 2,padding=7),
            #2000
            nn.Conv1d(10,15,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(15,15,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(15,15,5,padding=2),
            nn.Conv1d(15,15,5,stride = 2,padding=2),
            #1000
            nn.Conv1d(15,20,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(20,20,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(20,20,5,padding=2),
            nn.Conv1d(20,20,5,stride = 2,padding=2),
            #500
            nn.Conv1d(20,25,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(25,25,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(25,25,5,padding=2),
            nn.Conv1d(25,25,5,stride = 2,padding=7),
            #260
            nn.Conv1d(25,30,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(30,30,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(30,30,5,padding=2),
            nn.Conv1d(30,30,5,stride = 2,padding=7),
            #140
            nn.Conv1d(30,35,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(35,35,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(35,35,5,padding=2),
            nn.Conv1d(35,35,5,stride = 2,padding=7),
            #80
            nn.Conv1d(35,40,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(40,40,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(40,40,5,padding=2),
            nn.Conv1d(40,40,5,stride = 2,padding=2),
            #40
            nn.Conv1d(40,45,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(45,45,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(45,45,5,padding=2),
            nn.Conv1d(45,45,5,stride = 2,padding=2),
            #20
            nn.Conv1d(45,50,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(50,50,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(50,50,5,padding=2),
            nn.Conv1d(50,50,5,stride = 2,padding=2),
            #10
            nn.Conv1d(50,55,5,padding=2),
            #nn.LeakyReLU(),
            nn.Conv1d(55,55,5,padding = 2),
            #nn.LeakyReLU(),
            nn.Conv1d(55,55,5,padding = 2),
            nn.Conv1d(55,55,5,stride = 2),
            #3
            nn.Flatten(),
            nn.Linear(165,165),
            #nn.LeakyReLU(),
            nn.Linear(165,50),
            nn.Linear(50,len(hp.classification_thresholds)+1),
        ]
        for _layer in _layers:
            self.append(_layer)

