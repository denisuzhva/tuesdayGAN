import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


SAMPLE_RATE = 44100
BIT_DEPTH = 16
DATA_DIM = 128

def plotSpecGr(f, t, sg):
    _, (ax1, ax2) = plt.subplots(1, 2,sharey=True, figsize=(14, 6))
    ax1.pcolormesh(t, f, 10*np.log10(sg.real), cmap='inferno', label='Re')
    ax2.pcolormesh(t, f, 10*np.log10(sg.imag), cmap='inferno', label='Im')
    plt.show()

def makeSpecGr(read_path):
    tup_wav = wavfile.read(read_path)
    all_signal = tup_wav[1]
    assert tup_wav[0] == SAMPLE_RATE
    f, t, sg = signal.stft(all_signal[:, 0],
                           SAMPLE_RATE,
                           noverlap=0,
                           nperseg=DATA_DIM*2)
    sg = np.delete(sg, 0, 0)
    sg_mag = np.abs(sg) ** 2
    sg_mag = 10*np.log10(sg_mag + 1)
    sg_mag_max = np.max(sg_mag)
    sg_mag /= sg_mag_max
    sg_pha = np.angle(sg)
    sg_pha_min = np.min(sg_pha)
    sg_pha -= sg_pha_min
    sg_pha /= -sg_pha_min*2
    print(np.max(sg_mag))
    print(np.min(sg_mag))
    print(np.max(sg_pha))
    print(np.min(sg_pha))
    sg_conc = np.stack((sg_mag, sg_pha), axis=2).astype('float32')
    return sg_conc
   
def sliceSpecGr(sg_conc):
    n_data_samples = sg_conc.shape[1] // DATA_DIM
    data_set = np.zeros((n_data_samples, DATA_DIM, DATA_DIM, 2))
    for data_iter in range(n_data_samples):
        data_set[data_iter, :, :, :] = sg_conc[:, 
                                               data_iter*DATA_DIM:(data_iter+1)*DATA_DIM,
                                               :]
    return data_set

if __name__ == '__main__':
    #IMPORT
    #CONVERT
    #SLICE
    #WRITE

    read_path = '../../TGAN_DATASET/wav/anim1.wav'
    write_path = '../../TGAN_DATASET/anim1/full.npy'
    sg_conc = makeSpecGr(read_path)
    data_set = sliceSpecGr(sg_conc).astype('float32')
    print(data_set.shape)
    np.save(write_path, data_set)