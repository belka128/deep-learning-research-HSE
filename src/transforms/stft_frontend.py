import random

import torch
import torch.nn.functional as F
from torch import nn


class STFTFrontend(nn.Module):
    """
    STFT-based front-end + trim-pad data preparation, combining:

    - the front-end used in the FFT-LCNN system of the STC ASVspoof2019
      paper (Lavrentyeva et al., arXiv:1904.05576): raw log power magnitude
      spectrum, computed via FFT with a Blackman window (n_fft=1724,
      hop ~= 0.0081s, giving n_fft // 2 + 1 = 863 frequency bins, matching
      the input size of Table 1 in that paper);
    - the "trim-pad" fixed-length data preparation scheme of Wang &
      Yamagishi (arXiv:2103.11326): the feature sequence is either
      zero-padded (if shorter than K frames) or cropped to K frames
      starting at a (during training) random offset x_{n:n+K}.

    Used as an instance_transform on the "data_object" tensor (the raw
    waveform produced by ASVspoofDataset.load_object): converts a 1D
    waveform into a fixed-size (n_freq, n_frames) log power spectrogram.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=1724,
        hop_seconds=0.0081,
        n_frames=750,
        random_crop=True,
        eps=1e-6,
    ):
        """
        Args:
            sample_rate (int): audio sample rate.
            n_fft (int): FFT size (also used as the STFT window length).
            hop_seconds (float): hop length in seconds between STFT frames.
            n_frames (int): fixed number of output time frames (K).
            random_crop (bool): if True (train partition), crop a random
                K-frame window from utterances longer than K frames. If
                False (dev/eval partitions), deterministically take the
                first K frames, for reproducible scoring.
            eps (float): additive constant for numerical stability of log.
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = round(hop_seconds * sample_rate)
        self.n_frames = n_frames
        self.random_crop = random_crop
        self.eps = eps
        self.register_buffer("window", torch.blackman_window(n_fft), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform (Tensor): 1D raw waveform tensor.
        Returns:
            spec (Tensor): (n_freq, n_frames) log power spectrogram, with
                n_freq = n_fft // 2 + 1.
        """
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = stft.real.pow(2) + stft.imag.pow(2)
        log_power = torch.log(power + self.eps)

        n_freq, n_time = log_power.shape
        if n_time >= self.n_frames:
            start = (
                random.randint(0, n_time - self.n_frames)
                if self.random_crop
                else 0
            )
            log_power = log_power[:, start : start + self.n_frames]
        else:
            log_power = F.pad(log_power, (0, self.n_frames - n_time))

        return log_power
