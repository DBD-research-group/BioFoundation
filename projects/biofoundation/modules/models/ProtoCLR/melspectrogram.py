from torchaudio import transforms as T
import torch
import torch.nn as nn

MEAN, STD = 0.5347, 0.0772 # Xeno-Canto stats
SR = 16000
NFFT = 1024
HOPLEN = 320
NMELS = 128
FMIN = 50
FMAX = 8000
EMBEDDING_SIZE = 768

class Normalization(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
class Standardization(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()

        self.mean = mean
        self.std = std
    
    def forward(self, x):
        return (x - self.mean) / self.std

class MelSpectrogramProcessor(torch.nn.Module):
    def __init__(self, sample_rate=SR, n_mels=NMELS, n_fft=NFFT, hop_length=HOPLEN, f_min=FMIN, f_max=FMAX):
        super().__init__()
        self.transform = nn.Sequential(
            T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, f_min=f_min, f_max=f_max),
            T.AmplitudeToDB(),
            Normalization(),
            Standardization(mean=MEAN, std=STD),
        )
    def forward(self, waveform):
        return self.transform(waveform)