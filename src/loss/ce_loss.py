import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    """
    Standard Cross-Entropy loss for the bonafide/spoof classification head.

    We use plain Cross-Entropy instead of the angular-margin softmax
    (A-softmax) used in the original STC paper. Wang & Yamagishi
    (arXiv:2103.11326) show, on this exact ASVspoof2019 LA task, that the
    margin-based losses (A-softmax, AM-softmax, OC-softmax) give a
    statistically insignificant improvement over a simple sigmoid/CE
    objective for an LCNN back end (their Fig. 3 significance test), while
    adding extra margin hyperparameters that need tuning. Given this, and
    that our target EER budget is generous, Cross-Entropy is the simpler
    and more robust choice.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Args:
            logits (Tensor): model output of shape (batch, 2).
            labels (Tensor): ground-truth labels (0=spoof, 1=bonafide).
        Returns:
            losses (dict): dict with the "loss" key.
        """
        return {"loss": self.loss(logits, labels)}
