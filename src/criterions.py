import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, predicted: torch.Tensor(), target: torch.Tensor()):
        predicted = predicted.float()
        target = target.float()

        logprobs = F.log_softmax(predicted, dim=-1)
        nll_loss = -logprobs * target
        nll_loss = nll_loss.sum(-1)

        smooth_loss = -logprobs.mean(dim=-1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
