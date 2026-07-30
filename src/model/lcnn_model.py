import torch
from torch import nn

from src.model.mfm import MFM


def _conv_mfm(in_channels, out_channels, kernel_size, padding):
    """
    Conv2d -> MFM block. MFM halves out_channels, so the Conv2d must
    produce 2 * out_channels feature maps.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of channels AFTER the MFM activation.
        kernel_size (int): square convolution kernel size.
        padding (int): convolution padding (kernel_size // 2 for "same").
    Returns:
        block (nn.Sequential): Conv2d followed by MFM.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=1, padding=padding),
        MFM(mode="conv"),
    )


class LCNN(nn.Module):
    """
    LightCNN (LCNN) countermeasure system for voice anti-spoofing, following
    the "enhanced Light CNN" architecture described in Table 1 of the STC
    Antispoofing Systems paper for ASVspoof2019 (Lavrentyeva et al.,
    arXiv:1904.05576), which itself builds on the MFM activation of
    Wu et al. (arXiv:1511.02683).

    Input is a single-channel (log power spectrum) time-frequency
    representation of shape (batch, 1, n_freq, n_frames).

    Per the homework hint, a Dropout layer is placed before the final
    BatchNorm (STC paper reports dropout p=0.75).
    """

    def __init__(self, n_freq=863, n_frames=750, n_class=2, dropout=0.75):
        """
        Args:
            n_freq (int): number of frequency bins of the input spectrogram
                (STFT with n_fft=1724 gives n_fft // 2 + 1 = 863 bins).
            n_frames (int): fixed number of time frames of the input
                spectrogram (trim-pad length K).
            n_class (int): number of output classes (2: bonafide / spoof).
            dropout (float): dropout probability applied before the final
                BatchNorm1d, as in the original paper.
        """
        super().__init__()

        self.features = nn.Sequential(
            # Conv_1 + MFM_2
            _conv_mfm(1, 32, kernel_size=5, padding=2),
            # MaxPool_3
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv_4 + MFM_5
            _conv_mfm(32, 32, kernel_size=1, padding=0),
            # BatchNorm_6
            nn.BatchNorm2d(32),
            # Conv_7 + MFM_8
            _conv_mfm(32, 48, kernel_size=3, padding=1),
            # MaxPool_9
            nn.MaxPool2d(kernel_size=2, stride=2),
            # BatchNorm_10
            nn.BatchNorm2d(48),
            # Conv_11 + MFM_12
            _conv_mfm(48, 48, kernel_size=1, padding=0),
            # BatchNorm_13
            nn.BatchNorm2d(48),
            # Conv_14 + MFM_15
            _conv_mfm(48, 64, kernel_size=3, padding=1),
            # MaxPool_16
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv_17 + MFM_18
            _conv_mfm(64, 64, kernel_size=1, padding=0),
            # BatchNorm_19
            nn.BatchNorm2d(64),
            # Conv_20 + MFM_21
            _conv_mfm(64, 32, kernel_size=3, padding=1),
            # BatchNorm_22
            nn.BatchNorm2d(32),
            # Conv_23 + MFM_24
            _conv_mfm(32, 32, kernel_size=1, padding=0),
            # BatchNorm_25
            nn.BatchNorm2d(32),
            # Conv_26 + MFM_27
            _conv_mfm(32, 32, kernel_size=3, padding=1),
            # MaxPool_28
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        flatten_dim = self._infer_flatten_dim(n_freq, n_frames)

        self.classifier = nn.Sequential(
            # FC_29 + MFM_30 (Linear must output 2 * 80 for MFM to halve it to 80)
            nn.Linear(flatten_dim, 2 * 80),
            MFM(mode="linear"),
            # Dropout before the final BatchNorm, per the homework hint
            nn.Dropout(p=dropout),
            # BatchNorm_31
            nn.BatchNorm1d(80),
            # FC_32
            nn.Linear(80, n_class),
        )

        self._init_weights()

    @torch.no_grad()
    def _infer_flatten_dim(self, n_freq, n_frames):
        """
        Run a dummy forward pass through the conv stack to determine the
        flattened feature dimension for FC_29, avoiding manual arithmetic
        with 4 rounds of stride-2 max pooling.
        """
        dummy = torch.zeros(1, 1, n_freq, n_frames)
        out = self.features(dummy)
        return out.view(1, -1).shape[1]

    def _init_weights(self):
        """LCNN weights are initialized with (Kaiming) normal init, per the STC paper."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, data_object: torch.Tensor, **batch):
        """
        Args:
            data_object (Tensor): input spectrogram, shape
                (batch, n_freq, n_frames) or (batch, 1, n_freq, n_frames).
        Returns:
            output (dict): dict with "logits" of shape (batch, n_class).
        """
        x = data_object
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = x.flatten(1)
        logits = self.classifier(x)
        return {"logits": logits}

    def __str__(self):
        """Model prints with the number of parameters."""
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info
