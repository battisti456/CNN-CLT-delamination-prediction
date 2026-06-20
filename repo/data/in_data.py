import numpy as np
import scipy.signal
from run.hyperparameters import Hyperparameters

from common import SIGNAL_LENGTH, NUM_REPLICATES, ULTRASOUND_SAMPLE_RATE

from data._data import get_data

def _preprocess_signals(
        hp: Hyperparameters,
        signals: np.ndarray[
            tuple[int, int], np.dtype[np.float32]
        ],
) ->np.ndarray[
        tuple[int, int], np.dtype[np.float32]
    ]:
    #normalize the signals
    signals /= np.max(np.abs(signals),axis = -1)[...,np.newaxis]
    #apply low-pass filter
    b,a = scipy.signal.butter(#type:ignore
        N = hp.butter_order,
        Wn = hp.cutoff_frequency,
        fs = ULTRASOUND_SAMPLE_RATE,
        btype='low',
        analog=False
    )
    signals = scipy.signal.lfilter(
        b = b,
        a = a,
        x = signals,
        axis  = -1
    )#type:ignore
    return signals

def boost_signals(
        hp:Hyperparameters,
        signals:np.ndarray[
            tuple[int, int], np.dtype[np.float32]
        ]
) -> np.ndarray[
            tuple[int, int], np.dtype[np.float32]
        ]:
    if hp.in_interpolation_mode == 'none':
        return signals
    else:
        raise NotImplementedError()

def get_boosted_in_data(
        hp:Hyperparameters,
        indices:np.ndarray[tuple[int],np.dtype[np.intp]],
        nums:np.ndarray[tuple[int],np.dtype[np.intp]]|int = 1,
) -> np.ndarray[tuple[int],np.dtype[np.float32]]|np.ndarray[tuple[int,int],np.dtype[np.float32]]:
    if isinstance(nums,int):
        num = nums
        nums = np.zeros_like(indices)
        nums[:] = num
    all_signals = np.reshape(
        get_data('ultrasound',hp,axis=1),
        (
            NUM_REPLICATES,
            hp.num_blocks,
            SIGNAL_LENGTH
        )
    )
    length = np.sum(nums)
    to_return = np.empty((length,SIGNAL_LENGTH),np.float32)
    to_return[:] = np.nan
    write_index = 0
    for i in range(len(indices)):
        index = indices[i]
        num:int = nums[i]
        data = all_signals[:,index,:]
        data = data[~np.any(np.isnan(data),axis = -1),:]
        n = min(num,data.shape[0])
        to_return[write_index:write_index+n,:] = data[:n,:]
        write_index += n
        if num <= data.shape[0]:
            continue
        n = num - n
        boosted_indices= np.repeat(np.arange(data.shape[0]),np.ceil(n/data.shape[0]))
        np.random.shuffle(boosted_indices)
        boosted_indices = boosted_indices[:n]
        to_return[write_index:write_index+n,:] = boost_signals(hp,data[boosted_indices,:])
        write_index += n
    return _preprocess_signals(hp,to_return)