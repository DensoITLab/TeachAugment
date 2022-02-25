import torch
import torch.nn.functional as F


def non_saturating_loss(logits, targets):
    probs = logits.softmax(1)
    log_prob = torch.log(1 - probs + 1e-12)
    if targets.ndim == 2:
        return - (targets * log_prob).sum(1).mean()
    else:
        return F.nll_loss(log_prob, targets)


class NonSaturatingLoss(torch.nn.Module):
    def __init__(self, epsilon=0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        if self.epsilon > 0: # label smoothing
            n_classes = logits.shape[1]
            onehot_targets = F.one_hot(targets, n_classes).float()
            targets = (1 - self.epsilon) * onehot_targets + self.epsilon / n_classes
        return non_saturating_loss(logits, targets)
