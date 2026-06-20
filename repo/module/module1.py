import torch.nn as nn

from run.hyperparameters import Hyperparameters


class Module1(nn.Sequential):
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        self._hp = hp
        #7928
        self.pad0 = nn.ZeroPad1d(6)
        #7940
        self.conv0 = nn.Conv1d(1,5,5,padding=2)
        #self.lrelu01 = nn.LeakyReLU()
        self.conv1 = nn.Conv1d(5,5,5,stride=2,padding = 7)
        #3980
        self.conv2 = nn.Conv1d(5,10,5,padding=2)
        #self.lrelu23 = nn.LeakyReLU()
        self.conv3 = nn.Conv1d(10,10,5,stride = 2,padding=7)
        #2000
        self.conv4 = nn.Conv1d(10,15,5,padding=2)
        #self.lrelu45 = nn.LeakyReLU()
        self.conv5 = nn.Conv1d(15,15,5,stride = 2,padding=2)
        #1000
        self.conv6 = nn.Conv1d(15,20,5,padding=2)
        #self.lrelu67 = nn.LeakyReLU()
        self.conv7 = nn.Conv1d(20,20,5,stride = 2,padding=2)
        #500
        self.conv8 = nn.Conv1d(20,25,5,padding=2)
        #self.lrelu89 = nn.LeakyReLU()
        self.conv9 = nn.Conv1d(25,25,5,stride = 2,padding=7)
        #260
        self.conv10 = nn.Conv1d(25,30,5,padding=2)
        #self.lrelu1011 = nn.LeakyReLU()
        self.conv11 = nn.Conv1d(30,30,5,stride = 2,padding=7)
        #140
        self.conv12 = nn.Conv1d(30,35,5,padding=2)
        #self.lrelu1213 = nn.LeakyReLU()
        self.conv13 = nn.Conv1d(35,35,5,stride = 2,padding=7)
        #80
        self.conv14 = nn.Conv1d(35,40,5,padding=2)
        self.conv15 = nn.Conv1d(40,40,5,stride = 2,padding=2)
        #40
        self.conv16 = nn.Conv1d(40,45,5,padding=2)
        #self.lrelu1617 = nn.LeakyReLU()
        self.conv17 = nn.Conv1d(45,45,5,stride = 2,padding=2)
        #20
        self.conv18 = nn.Conv1d(45,50,5,padding=2)
        #self.lrelu1819 = nn.LeakyReLU()
        self.conv19 = nn.Conv1d(50,50,5,stride = 2,padding=2)
        #10
        self.conv20 = nn.Conv1d(50,55,5,padding=2)
        #self.lrelu2021 = nn.LeakyReLU()
        self.conv21 = nn.Conv1d(55,55,5,stride = 2)
        #3
        self.flatten = nn.Flatten()
        self.full0 = nn.Linear(165,165)
        self.full1 = nn.Linear(165,len(hp.classification_thresholds)+1)
        # self.full1 = nn.Linear(
        #     165,
        #     1 if hp.out_shape == 'block' else
        #     2 if hp.out_shape == 'layer' else
        #     3
        # )
        #self.sigmoid = nn.Sigmoid()

