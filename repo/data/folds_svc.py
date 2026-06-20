from typing import Generator

import torch
import torch.utils.data

from run.hyperparameters import Hyperparameters

from common import NUM_REPLICATES, NUM_X, NUM_BLOCKS, NUM_LAYUPS
import numpy as np
from data.out_data import get_boosted_out_data
from data.in_data import get_boosted_in_data, _preprocess_signals
from data.misc import class_freq, get_found ,spectrogram
from dataclasses import replace
import random
import scipy.signal
from data._data import _data
from data.misc import classify
from common import SIGNAL_LENGTH

def get_boosted_vmd_data(
        hp:Hyperparameters,
        indices:np.ndarray[tuple[int],np.dtype[np.intp]],
        nums:np.ndarray[tuple[int],np.dtype[np.intp]]|int = 1,
        append_poster:bool = False
) -> tuple[np.ndarray[tuple[int,int],np.dtype[np.float32]],np.ndarray[tuple[int,int],np.dtype[np.float32]]]:
    if isinstance(nums,int):
        num = nums
        nums = np.zeros_like(indices)
        nums[:] = num
    data_3ply = np.load('data\\vmd_3ply.npz')
    data_5ply = np.load('data\\vmd_5ply.npz')
    if hp.ply == '3ply':
        signal = data_3ply['decomposed_signals']
        mcfs = data_3ply['mode_center_frequencies']
    elif hp.ply == '5ply':
        signal = data_5ply['decomposed_signals']
        mcfs = data_5ply['mode_center_frequencies']
    elif hp.ply == 'both':
        signal = np.concatenate((data_3ply['decomposed_signals'],data_5ply['decomposed_signals']),axis=0)
        mcfs = np.concatenate((data_3ply['mode_center_frequencies'],data_5ply['mode_center_frequencies']),axis=0)
    else:
        raise NotImplementedError()
    signal = np.moveaxis(signal,(0,1,2,3,4),(1,0,2,3,4)).reshape((NUM_REPLICATES,hp.num_blocks,12,SIGNAL_LENGTH))
    mcfs = np.moveaxis(mcfs,(0,1,2,3),(1,0,2,3)).reshape((NUM_REPLICATES,hp.num_blocks,12))

    length = np.sum(nums)
    to_return_rmsv = np.empty((length,12),np.float32)
    to_return_mcf = np.empty((length,12),np.float32)
    to_return_rmsv[:] = np.nan
    to_return_mcf[:] = np.nan
    write_index = 0
    for i in range(len(indices)):
        index = indices[i]
        num:int = nums[i]
        rmsv = np.sqrt(np.mean(signal[:,index,:,:]**2, axis = -1))
        mcf = mcfs[:,index,:]
        fail_nan = np.logical_or(np.any(np.isnan(rmsv), axis = -1),np.any(np.isnan(mcf), axis = -1))
        rmsv = rmsv[~fail_nan,:]
        mcf = mcf[~fail_nan,:]
        n = min(num,rmsv.shape[0])
        to_return_rmsv[write_index:write_index+n,:] = rmsv[:n,:]
        to_return_mcf[write_index:write_index+n,:] = mcf[:n,:]
        write_index += n
        n = num - n
        if n <=0 :
            continue
        boosted_indices= np.repeat(np.arange(rmsv.shape[0]),np.ceil(n/rmsv.shape[0]))
        np.random.shuffle(boosted_indices)
        boosted_indices = boosted_indices[:n]
        to_return_rmsv[write_index:write_index+n,:] = rmsv[boosted_indices,:]
        to_return_mcf[write_index:write_index+n,:] = mcf[boosted_indices,:]
        write_index += n
    if append_poster:
        data_poster = np.load('data\\vmd_poster.npz')
        signal = data_poster['decomposed_signals']
        mcfs = data_poster['mode_center_frequencies']
        rmsv = np.sqrt(np.mean(signal**2, axis = -1))
        rmsv = rmsv.reshape((mcfs.shape[0]*mcfs.shape[1],12))
        mcfs = mcfs.reshape((mcfs.shape[0]*mcfs.shape[1],12))
        to_return_rmsv = np.concatenate((to_return_rmsv,rmsv),axis = 0)
        to_return_mcf = np.concatenate((to_return_mcf,mcfs),axis = 0)
    return to_return_rmsv,to_return_mcf


class Data(torch.utils.data.Dataset):
    def __init__(
            self,
            hp:Hyperparameters,
            base_indices:np.ndarray[tuple[int],np.dtype[np.intp]],
            extend_by:np.ndarray[tuple[int],np.dtype[np.intp]]|int,
            append_poster:bool = False
        ):
        super().__init__()
        self.hp = hp
        self.base_indices = base_indices
        num_layers = np.empty_like(self.base_indices)
        if hp.ply in ('3ply','5ply'):
            num_layers[:] = hp.num_layers
        elif hp.ply == 'both':
            num_layers[self.base_indices >= NUM_BLOCKS] = 4#5ply
            num_layers[self.base_indices < NUM_BLOCKS] = 2#3ply
        else:
            raise NotImplementedError()
        self.num_layers = np.repeat(num_layers,extend_by)
        self.labels = get_boosted_out_data(hp,base_indices,extend_by)
        self.rmsv,self.mcfs = get_boosted_vmd_data(hp,base_indices,extend_by,append_poster)
        if append_poster:
            labels = _data['block_edge_delams_poster'].flatten()
            labels = np.repeat(labels,5)
            if hp.classify:
                labels = classify(labels,hp.classification_thresholds)
            num_layers = np.empty_like(labels)
            num_layers[:] = 2
            self.labels = np.concat((self.labels,labels), axis = 0)
            self.num_layers = np.concat((self.num_layers,num_layers), axis = 0)

def get_folds(hp:Hyperparameters) -> Generator[tuple[
        Data,
        Data,
    ]|None,None,None]:
    """Returns datasets in pairs of train, test for k-fold testing

    Args:
        num_folds: the number of folds to divide into, or k
        out_data: what is the output parameter, should be based on the self.layer... and self.block... parameters

    Yields:
        An iterator going over every fold split for the data into training and testing
    """
    hp_no_noise = replace(hp,out_boost_amp=0)
    base_indices = np.arange(hp.num_blocks)
    classified_unboosted = get_boosted_out_data(replace(hp_no_noise,classify=True),base_indices)
    if hp.reorder_layups == 'auto':
        classified_unboosted_by_layup = np.reshape(classified_unboosted,(hp.num_layups,NUM_X))
        all_pass = np.all(classified_unboosted_by_layup == 0,axis = -1)
        all_fail = np.all(classified_unboosted_by_layup == 1,axis = -1)
        mixed = (~all_pass & ~all_fail)
        orders:tuple[list[int]]|tuple[list[int],list[int]] = (#type:ignore
            (list(range(NUM_LAYUPS)),) if hp.ply in ('3ply','5ply') 
            else (list(range(NUM_LAYUPS)),list(range(NUM_LAYUPS,NUM_LAYUPS*2)),))
        for order in orders:#only works for 6 fold...., not sure what I was thinking
            random.shuffle(order)
            for i in range(0,12,2):
                if all_pass[i]:#want all fail, or mixed
                    found = get_found(i,order,all_fail,mixed)
                elif all_fail[i]:#want all pass, or mixed
                    found = get_found(i,order,all_pass,mixed)
                else:#mixes
                    found = get_found(i,order,all_pass,all_fail)
                temp = order[found]
                order[found] = order[i+1]
                order[i+1] = temp
        hp.reorder_layups = tuple(np.array(orders).T.flatten().astype(object))
    if isinstance(hp.reorder_layups,tuple):
        layup_indices = base_indices.reshape((hp.num_layups,NUM_X))
        layup_indices = layup_indices[hp.reorder_layups,:]
        base_indices = layup_indices.flatten()
    fold_size = int(hp.num_blocks/hp.num_folds)
    if hp.ignore_proper_folds:
        np.random.shuffle(base_indices)
    for fold in range(hp.num_folds):
        start = fold*fold_size
        end = (fold+1)*fold_size
        train_indices = base_indices[np.r_[0:start,end:hp.num_blocks]]
        test_indices = base_indices[start:end]
        classified_unboosted = get_boosted_out_data(replace(hp_no_noise,classify=True),base_indices)
        class_freqs_unboosted_train = class_freq(classified_unboosted[train_indices],hp.num_classes)#type:ignore
        class_freqs_unboosted_test = class_freq(classified_unboosted[test_indices],hp.num_classes)#type:ignore
        extend_by:np.ndarray[tuple[int],np.dtype[np.intp]]
        if hp.skip_incomplete_folds and (np.all(class_freqs_unboosted_train == 0) or np.all(class_freqs_unboosted_test==0)):
            yield None
            continue
        if hp.data_balance_mode == 'balance':
            if hp.out_shape != 'block':
                raise NotImplementedError()
            #class_weights = 1/(class_freqs/class_freqs.sum())**0.5
            class_weights = 1/(class_freqs_unboosted_train/class_freqs_unboosted_train.sum())
            class_extend_by = class_weights/np.min(class_weights)*NUM_REPLICATES
            extend_by = np.zeros_like(train_indices)
            for i in range(hp.num_classes):
                index = classified_unboosted[train_indices] == i
                low = np.floor(class_extend_by[i])
                high = np.ceil(class_extend_by[i])
                num_high = int((class_extend_by[i]-low)*class_freqs_unboosted_train[i])
                extend_by_ = np.empty(class_freqs_unboosted_train[i],np.intp)
                extend_by_[:num_high] = high
                extend_by_[num_high:] = low
                np.random.shuffle(extend_by_)
                extend_by[index] = extend_by_
        elif hp.data_balance_mode == 'match':
            extend_by = np.empty_like(train_indices)
            extend_by[:] = NUM_REPLICATES
        else:
            raise NotImplementedError()
        yield (
            Data(hp,train_indices,extend_by, append_poster=hp.append_poster_to_train and hp.out_type == 'edge' and hp.out_shape == 'block'),
            Data(hp,test_indices,NUM_REPLICATES)
        )