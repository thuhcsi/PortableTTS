import librosa
import torch
from torch import nn

from feature import spectrogram


class MelspectrogramLayer(nn.Module):

    def __init__(self, sr, n_fft, hop_size, win_size, n_mels, fmin, fmax=None):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        # mel_basis: (n_mel, n_fft+1)
        self.mel_basis = nn.Parameter(torch.from_numpy(
            librosa.filters.mel(sr=sr,
                                n_fft=n_fft,
                                n_mels=n_mels,
                                fmin=fmin,
                                fmax=fmax)).unsqueeze(0),
                                      requires_grad=False)
        self.window = nn.Parameter(torch.hann_window(win_size),
                                   requires_grad=False)

    def forward(self, x):
        """Calculating melspectrogram using torch.
        Args:
            x (torch.Tensor): input wav signal of shape (b,t).
        Returns:
            torch.Tensor: melspectrogram of shape (b,n_mels,num_frames)
        """
        raw_spec = torch.abs(
            spectrogram.torch_stft(x, self.n_fft, self.hop_size, self.win_size,
                                   self.window))
        mel = torch.log(torch.clamp(torch.matmul(self.mel_basis, raw_spec),min=1e-5))
        return mel