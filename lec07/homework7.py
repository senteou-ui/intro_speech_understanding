import numpy as np

def major_chord(f, Fs):
    N = int(Fs * 0.5)
    n = np.arange(N)
    f2 = f * (2 ** (4/12))
    f3 = f * (2 ** (7/12))
    x = np.cos(2*np.pi*f*n/Fs) + np.cos(2*np.pi*f2*n/Fs) + np.cos(2*np.pi*f3*n/Fs)
    return x

def dft_matrix(N):
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-1j * 2 * np.pi * k * n / N)
    return W

def spectral_analysis(x, Fs):
    N = len(x)
    X = np.fft.fft(x)
    mag = np.abs(X)
    half_N = N // 2
    mag_half = mag[:half_N]
    indices = np.argsort(mag_half)[-3:]
    freqs = np.sort(indices * Fs / N)
    return freqs[0], freqs[1], freqs[2]