import numpy as np
from run.hyperparameters import Hyperparameters
import scipy.signal

def classify(arr:np.ndarray,classification_thresholds:tuple[float,...]) -> np.ndarray:
    to_return = np.zeros_like(arr,dtype=np.int64)
    c = (0,) + classification_thresholds  + (1,)
    for i in range(1,len(classification_thresholds)+1):
        above_min = arr >= c[i]
        if i == len(classification_thresholds):
            below_max = arr <= c[i+1]
        else:
            below_max = arr < c[i+1]
        to_return[np.logical_and(above_min,below_max)] = i
    return to_return


def class_freq(data:np.ndarray[tuple[int],np.dtype[np.intp]],num_classes:int) -> np.ndarray[tuple[int],np.dtype[np.intp]]:
    to_return = np.zeros(num_classes,np.intp)
    for j in range(num_classes):
            to_return[j] += (data == j).sum()
    return to_return

def get_first(arr:np.ndarray[tuple[int],np.dtype[np.bool]]) -> None|np.intp:
    i = np.argmax(arr)
    if i == 0 and arr[i] == 0:
        return None
    else:
        return i

def get_found(i,order,arr1,arr2) -> int:
    found = get_first(arr1[order[i+1:]])
    if found is None:
        found = get_first(arr2[order[i+1:]])
    if found is None:
        found = i+1
    else:
        found += i + 1
    return found

def spectrogram(hp:Hyperparameters,signal):
    return np.log(scipy.signal.spectrogram(
        signal,
        nperseg=hp.spectrogram_config['nperseg'],
        noverlap=hp.spectrogram_config['noverlap'],
        nfft=hp.spectrogram_config['nfft'],
        mode='magnitude',
        axis = -1
    )[2][:,:hp.spectrogram_config['width'],:hp.spectrogram_config['height']])

def fft(hp:Hyperparameters,signal):
    fft =  np.fft.rfft(signal,axis = -1)
    fft = np.moveaxis(np.array(
        [np.real(fft),np.imag(fft)]
    ),(0,1,2),(1,0,2))
    return fft