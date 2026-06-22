import numpy as np

def VAD(waveform, Fs):
    frame_length = int(0.025 * Fs)
    step = int(0.01 * Fs)
    num_frames = int((len(waveform) - frame_length) / step) + 1
    
    energies = np.zeros(num_frames)
    for i in range(num_frames):
        frame = waveform[i * step : i * step + frame_length]
        energies[i] = np.sum(frame ** 2)
        
    threshold = 0.1 * np.max(energies)
    is_active = energies > threshold
    
    segments = []
    in_segment = False
    start_idx = 0
    
    for i in range(num_frames):
        if is_active[i] and not in_segment:
            start_idx = i
            in_segment = True
        elif not is_active[i] and in_segment:
            end_idx = i - 1
            start_sample = start_idx * step
            end_sample = end_idx * step + frame_length
            segments.append(waveform[start_sample:end_sample])
            in_segment = False
            
    if in_segment:
        start_sample = start_idx * step
        end_sample = (num_frames - 1) * step + frame_length
        segments.append(waveform[start_sample:end_sample])
        
    return segments

def segments_to_models(segments, Fs):
    models = []
    frame_length = int(0.004 * Fs)
    step = int(0.002 * Fs)
    half_N = frame_length // 2
    
    for seg in segments:
        pre = np.append(seg[0], seg[1:] - 0.97 * seg[:-1])
        num_frames = int((len(pre) - frame_length) / step) + 1
        specs = []
        for i in range(num_frames):
            frame = pre[i * step : i * step + frame_length]
            mag = np.abs(np.fft.fft(frame))[:half_N]
            specs.append(np.log(np.maximum(mag, 1e-10)))
        
        if specs:
            models.append(np.mean(specs, axis=0))
        else:
            models.append(np.zeros(half_N))
            
    return models

def recognize_speech(testspeech, Fs, models, labels):
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)
    
    Y = len(models)
    K = len(test_models)
    sims = np.zeros((Y, K))
    test_outputs = []
    
    for k, t_model in enumerate(test_models):
        for y, m_model in enumerate(models):
            dot_product = np.dot(m_model, t_model)
            norm_m = np.linalg.norm(m_model)
            norm_t = np.linalg.norm(t_model)
            if norm_m > 0 and norm_t > 0:
                sims[y, k] = dot_product / (norm_m * norm_t)
            else:
                sims[y, k] = 0.0
        
        best_idx = np.argmax(sims[:, k])
        test_outputs.append(labels[best_idx])
        
    return sims, test_outputs