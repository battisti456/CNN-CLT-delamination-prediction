import numpy as np
import torch
import torch.utils.data

from data.in_data import _preprocess_signals
from run.hyperparameters import Hyperparameters
from common import SIGNAL_LENGTH
import scipy.signal
from data.misc import spectrogram

def transform(hp:Hyperparameters,signal:np.ndarray[tuple[int,int],np.dtype[np.float32]]):
    length = signal.shape[0]
    if hp.transfer_learning_transform == 'spec_rect_mask':
        total_area = hp.spectrogram_config['width']*hp.spectrogram_config['height']
        goal_area = total_area*hp.transfer_learning_transform_config['portion_to_cover']
        to_return = signal.copy()
        max_width:int = min(hp.spectrogram_config['width'],int(goal_area))
        rand_width = np.random.random(length)*max_width
        rand_height = np.floor(goal_area/rand_width).astype(np.intp)
        rand_width = np.floor(rand_width).astype(np.intp)
        rand_x = np.floor(np.random.random(length)*(hp.spectrogram_config['width']-rand_width)).astype(np.intp)
        rand_y = np.floor(np.random.random(length)*(hp.spectrogram_config['height']-rand_height)).astype(np.intp)
        for i in range(length):
            to_return[i,rand_x[i]:rand_x[i]+rand_width[i],rand_y[i]:rand_y[i]+rand_height[i]] = 0
    elif hp.transfer_learning_transform == 'waveform_in_painting':
        width = np.random.randint(
            hp.transfer_learning_transform_config['min_width'],
            hp.transfer_learning_transform_config['max_width'],
            signal.shape[0]
        )
        pos = np.random.randint(
            0,
            SIGNAL_LENGTH-hp.transfer_learning_transform_config['max_width'],
            signal.shape[0]
        )
        to_return = signal.copy()
        for i in range(to_return.shape[0]):
            to_return[i,pos[i]:pos[i]+width[i]] = 0
    else:
        raise NotImplementedError()
    return to_return


class Transfer_D2(torch.utils.data.Dataset):
    def __init__(self,hp:Hyperparameters):
        super().__init__()
        self.hp = hp
        self.signal = np.repeat(
            _preprocess_signals(hp,np.load('data/transfer.npz')['transfer']),
            hp.transfer_learning_boost,
            axis=0
        )
    def __getitem__(self,idx):
        signal = self.signal[idx,:][None,...]
        if 'spectrogram' in self.hp.use_parameters:
            return (
                torch.tensor(transform(self.hp,spectrogram(self.hp,signal)),dtype = torch.float32),
                torch.tensor(spectrogram(self.hp,signal),dtype = torch.float32)
            )
        raise NotImplementedError()
    def __len__(self) -> int:
        return self.signal.shape[0]