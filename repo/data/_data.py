import numpy as np
from run.hyperparameters import Hyperparameters

_data:dict[str,np.ndarray] = np.load('data/data.npz')

def get_data(
        base_str:str,
        hp:Hyperparameters,
        axis = 0
):
    if hp.ply in ('3ply','5ply'):
        return _data[base_str+'_'+hp.ply]
    elif hp.ply == 'both':
        return np.concatenate(
            (_data[base_str+'_3ply'],_data[base_str+'_5ply']),
            axis=axis
        )
    else:
        raise NotImplementedError()