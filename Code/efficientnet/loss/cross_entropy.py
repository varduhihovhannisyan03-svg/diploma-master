from torch import Tensor, nn
from torch.nn import functional as F


class CrossEntropyLoss(nn.Module):
    """ NLL loss with label smoothing """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        probs = self.softmax(x)
        smooth_loss = -probs.mean(dim=-1)
        nll_loss = -probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
