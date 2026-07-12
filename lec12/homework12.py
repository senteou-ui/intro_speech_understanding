import numpy as np

def voiced_excitation(duration, F0, Fs):
    T0 = int(np.round(Fs / F0))
    excitation = np.zeros(duration)
    excitation[::T0] = -1
    return excitation

def resonator(x, F, BW, Fs):
    C = -np.exp(-2 * np.pi * BW / Fs)
    B = 2 * np.exp(-np.pi * BW / Fs) * np.cos(2 * np.pi * F / Fs)
    A = 1 - B - C
    
    y = np.zeros(len(x))
    if len(x) > 0:
        y[0] = A * x[0]
    if len(x) > 1:
        y[1] = A * x[1] + B * y[0]
        
    for n in range(2, len(y)):
        y[n] = A * x[n] + B * y[n-1] + C * y[n-2]
        
    return y

def synthesize_vowel(duration, F0, F1, F2, F3, F4, BW1, BW2, BW3, BW4, Fs):
    excitation = voiced_excitation(duration, F0, Fs)
    y1 = resonator(excitation, F1, BW1, Fs)
    y2 = resonator(y1, F2, BW2, Fs)
    y3 = resonator(y2, F3, BW3, Fs)
    y4 = resonator(y3, F4, BW4, Fs)
    
    return y4