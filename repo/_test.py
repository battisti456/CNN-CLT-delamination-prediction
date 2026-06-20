import numpy as np
import scipy.signal

data = np.load('data/transfer.npz')['transfer'][:50,:]

import matplotlib.pyplot as plt
f,t, spec = scipy.signal.spectrogram(
    x = data,
    fs = 0.1e-6,
    nperseg=120,
    noverlap=59,
    nfft=512,
    mode='magnitude',
    axis = -1
)
#spec = spec[:,:256,:128]
spec = np.log(spec)
print(np.min(spec),np.max(spec))

plt.imshow(spec[0,:])
plt.show()