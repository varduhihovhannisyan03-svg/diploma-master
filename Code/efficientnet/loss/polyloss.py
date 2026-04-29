import torch


class PolyLoss:
    """ [https://arxiv.org/abs/2204.12511] """

    def __init__(self, reduction='none', label_smoothing=0.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.softmax = torch.nn.Softmax(dim=-1)

    def __call__(self, prediction, target, epsilon=1.0):
        ce = F.cross_entropy(prediction, target, reduction=self.reduction, label_smoothing=self.label_smoothing)
        pt = torch.sum(F.one_hot(target, num_classes=1000) * self.softmax(prediction), dim=-1)
        pl = torch.mean(ce + epsilon * (1 - pt))
        return pl
