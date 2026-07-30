import torch
from torch import nn


class MFM(nn.Module):
    """
    Max-Feature-Map (MFM) activation, as used in the LightCNN architecture
    (Wu et al., "A Light CNN for Deep Face Representation", arXiv:1511.02683).

    Splits the input into two equally-sized halves along the channel
    (Conv2d input) or feature (Linear input) axis and takes the element-wise
    maximum of the two halves. This halves the number of channels/features
    and acts as a competitive feature selector between the two halves.
    """

    def __init__(self, mode: str = "conv"):
        """
        Args:
            mode (str): "conv" to split along the channel axis of a
                (N, C, H, W) tensor, "linear" to split along the feature
                axis of a (N, F) tensor.
        """
        super().__init__()
        assert mode in ("conv", "linear")
        self.split_dim = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): input tensor with an even-sized channel/feature axis.
        Returns:
            out (Tensor): element-wise max of the two halves of x.
        """
        left, right = x.chunk(2, dim=self.split_dim)
        return torch.maximum(left, right)
