import numpy as np
import torch, torch.nn

def get_features(waveform, Fs):
    vad_frame_len = int(0.025 * Fs)
    vad_step = int(0.01 * Fs)
    vad_num_frames = int((len(waveform) - vad_frame_len) / vad_step) + 1
    
    energies = np.zeros(vad_num_frames)
    for i in range(vad_num_frames):
        frame = waveform[i * vad_step : i * vad_step + vad_frame_len]
        energies[i] = np.sum(frame ** 2)
        
    threshold = 0.1 * np.max(energies)
    is_active = energies > threshold
    
    segments = []
    in_segment = False
    start_idx = 0
    for i in range(vad_num_frames):
        if is_active[i] and not in_segment:
            start_idx = i
            in_segment = True
        elif not is_active[i] and in_segment:
            end_idx = i - 1
            segments.append((start_idx * vad_step, end_idx * vad_step + vad_frame_len))
            in_segment = False
    if in_segment:
        segments.append((start_idx * vad_step, (vad_num_frames - 1) * vad_step + vad_frame_len))
        
    pre = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])
    feat_frame_len = int(0.004 * Fs)
    feat_step = int(0.002 * Fs)
    num_feat_frames = int((len(pre) - feat_frame_len) / feat_step) + 1
    half_N = feat_frame_len // 2
    
    features = np.zeros((num_feat_frames, half_N))
    for i in range(num_feat_frames):
        frame = pre[i * feat_step : i * feat_step + feat_frame_len]
        features[i] = np.abs(np.fft.fft(frame))[:half_N]
        
    labels = np.zeros(num_feat_frames, dtype=int)
    
    # 【關鍵修改】: 強制將找到的段落，等比例平均分配為 1 到 5 的標籤
    if len(segments) > 0:
        for seg_idx, (start, end) in enumerate(segments):
            label = (seg_idx * 5) // len(segments) + 1
            for i in range(num_feat_frames):
                center = i * feat_step + feat_frame_len / 2
                if start <= center <= end:
                    labels[i] = label
                
    return features, labels

def train_neuralnet(features, labels, iterations):
    NFEATS = features.shape[1]
    NLABELS = int(np.max(labels)) + 1
    
    model = torch.nn.Sequential(
        torch.nn.LayerNorm(NFEATS),
        torch.nn.Linear(NFEATS, NLABELS)
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    X = torch.tensor(features, dtype=torch.float32)
    Y = torch.tensor(labels, dtype=torch.long)
    
    lossvalues = np.zeros(iterations)
    
    for i in range(iterations):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        lossvalues[i] = loss.item()
        
    return model, lossvalues

def test_neuralnet(model, features):
    X = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.detach().numpy()