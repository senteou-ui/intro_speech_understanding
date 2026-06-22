import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    num_frames = int((len(waveform) - frame_length) / step) + 1

    frames = np.zeros((num_frames, frame_length))

    for i in range(num_frames):
        start = i * step
        frames[i] = waveform[start:start+frame_length]

    return frames


def frames_to_mstft(frames):
    return np.abs(np.fft.fft(frames, axis=1))


def mstft_to_spectrogram(mstft):
    floor = 0.001 * np.amax(mstft)

    spectrogram = 20 * np.log10(
        np.maximum(floor, mstft)
    )

    return spectrogram