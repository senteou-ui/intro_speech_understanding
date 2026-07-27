import numpy as np
import librosa

def lpc(speech, frame_length, frame_skip, order):
    """
    Perform linear predictive analysis of input speech.
    """
    nframes = int((len(speech) - frame_length) / frame_skip)
    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))
    
    for m in range(nframes):
        # 取出目前的 frame
        frame = speech[m * frame_skip : m * frame_skip + frame_length]
        
        # 計算 LPC 係數
        a = librosa.lpc(y=frame, order=order)
        A[m, :] = a
        
        # 計算激發訊號 (excitation)，即預測誤差，透過與 LPC 係數卷積取得
        excitation[m, :] = np.convolve(frame, a, mode='full')[:frame_length]
        
    return A, excitation

def synthesize(e, A, frame_skip):
    """
    Synthesize speech from LPC residual and coefficients.
    """
    nframes = A.shape[0]
    order = A.shape[1] - 1
    synthesis = np.zeros(len(e))
    
   
    for m in range(nframes):
        start = m * frame_skip
        end = (m + 1) * frame_skip
        for n in range(start, end):
            synthesis[n] = e[n]
            for k in range(1, order + 1):
                if n - k >= 0:
                    synthesis[n] -= A[m, k] * synthesis[n - k]
                    
    return synthesis

def robot_voice(excitation, T0, frame_skip):
    """
    Calculate the gain for each excitation frame, then create the excitation for a robot voice.
    """
    nframes = excitation.shape[0]
    gain = np.zeros(nframes)
    e_robot = np.zeros(nframes * frame_skip)
    
   
    for m in range(nframes):
      
        valid_exc = excitation[m, -frame_skip:]
        gain[m] = np.sqrt(np.mean(valid_exc**2))
        
  
    for n in range(len(e_robot)):
        if n % T0 == 0:
            m = n // frame_skip
       
            e_robot[n] = gain[m] * np.sqrt(T0)
            
    return gain, e_robot