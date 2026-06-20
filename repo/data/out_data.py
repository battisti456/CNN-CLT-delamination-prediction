from typing import assert_never

import numpy as np

from common import EDGE_STD, NUM_BLOCKS, NUM_SIDES
from data._data import _data
from run.hyperparameters import Hyperparameters
from data.misc import classify
from dataclasses import replace


def get_boosted_out_data(
        hp:Hyperparameters,
        indices:np.ndarray[tuple[int],np.dtype[np.intp]],
        nums:np.ndarray[tuple[int],np.dtype[np.intp]]|int = 1,
) -> np.ndarray[tuple[int],np.dtype[np.float32]]|np.ndarray[tuple[int,int],np.dtype[np.float32]]:
    rindices = np.repeat(indices,nums)
    data_func = (
        get_boosted_out_data_area if hp.out_type == 'area' else get_boosted_out_data_edge
    )
    if hp.ply == 'both':
        filter_3ply =  rindices<NUM_BLOCKS
        filter_5ply = rindices>=NUM_BLOCKS
        data_3ply = data_func(
            rindices[filter_3ply],
            replace(hp,ply='3ply'),
        )
        data_5ply = data_func(
            rindices[filter_5ply]-NUM_BLOCKS,
            replace(hp,ply='5ply'),
        )
        to_return = np.empty(rindices.shape,dtype=np.float32)
        to_return[filter_3ply] = data_3ply
        to_return[filter_5ply] = data_5ply
    else:
        to_return =  data_func(rindices,hp)
    if hp.classify:
        to_return = classify(to_return,hp.classification_thresholds)
    return to_return
def get_boosted_out_data_area(
        indices:np.ndarray[tuple[int],np.dtype[np.intp]],
        hp:Hyperparameters,
) -> np.ndarray[tuple[int],np.dtype[np.float32]]|np.ndarray[tuple[int,int],np.dtype[np.float32]]:
    layer_area_delams = np.reshape(_data[f'layer_area_delams_{hp.ply}'],(NUM_BLOCKS,hp.num_layers))[indices,:]
    layer_areas = np.reshape(_data[f'layer_areas_{hp.ply}'],(NUM_BLOCKS,hp.num_layers))[indices,:]
    if hp.out_boost_amp != 0:
        layer_area_delams = np.clip(a = np.random.normal(
                loc = layer_area_delams,
                #from out\plots\default_plots\area_delam_deviation\area_delam_deviation_3ply.svg
                scale = np.clip(-0.8*layer_area_delams**2+0.78*layer_area_delams,0.001,None)*hp.out_boost_amp
            ),
            a_min=0,
            a_max=1
        )
    if hp.out_shape == 'layer':
        return layer_area_delams#type:ignore
    block_area_delams = np.average(layer_area_delams,axis=-1,weights=layer_areas)
    if hp.out_shape == 'block':
        return block_area_delams#type:ignore
    else:
        assert_never(hp.out_shape)
def get_boosted_out_data_edge(
        indices:np.ndarray[tuple[int],np.dtype[np.intp]],
        hp:Hyperparameters,
) -> np.ndarray[tuple[int],np.dtype[np.float32]]|np.ndarray[tuple[int,int],np.dtype[np.float32]]:
    layer_edge_delam_lengths = np.reshape(
        _data[f'layer_edge_delam_lengths_{hp.ply}'],
        (NUM_BLOCKS,hp.num_layers,NUM_SIDES)
    )[indices,:,:]
    layer_edge_lengths = np.reshape(
        _data[f'layer_edge_lengths_{hp.ply}'],
        (NUM_BLOCKS,hp.num_layers,NUM_SIDES)
    )[indices,:,:]
    if hp.out_boost_amp != 0:
        layer_edge_delam_lengths = np.random.normal(
            loc = layer_edge_delam_lengths,
            scale=EDGE_STD*hp.out_boost_amp
        )
        layer_edge_lengths = np.random.normal(
            loc = layer_edge_lengths,
            scale=EDGE_STD*hp.out_boost_amp
        )
    layer_edge_delams = np.clip(
        np.sum(
            layer_edge_delam_lengths,
            axis = -1
        )/np.sum(
            layer_edge_lengths,
            axis = -1
        ),
        0,1
    )
    if hp.out_shape == 'layer':
        return layer_edge_delams#type:ignore
    block_edge_delams = np.clip(
        np.sum(
            layer_edge_delam_lengths,
            axis = (-1,-2)
        )/np.sum(
            layer_edge_lengths,
            axis = (-1,-2)
        ),
        0,1
    )
    if hp.out_shape == 'block':
        return block_edge_delams#type:ignore
    else:
        assert_never(hp.out_shape)