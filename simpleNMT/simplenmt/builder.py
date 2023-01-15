import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable

logger = logging.getLogger(__name__)

def build_optimizer(config, parameters):
    """创建 Adam optimizer
    """
    optimizer_name = config.get("optimizer", "sgd").lower()

    kwargs = {
        "lr": config.get("learning_rate", 3.0e-4),
        "weight_decay": config.get("weight_decay", 0),
    }

    kwargs["betas"] = config.get("adam_betas", (0.9, 0.999))
    optimizer = torch.optim.Adam(parameters, **kwargs)

    logger.info(
        "%s(%s)",
        optimizer.__class__.__name__,
        ", ".join([f"{k}={v}" for k, v in kwargs.items()]),
    )
    return optimizer


def build_criterion(label_smoothing, pad_index):
    criterion = XentLoss(smoothing=label_smoothing, pad_index=pad_index)
    logger.info(f"{criterion.__class__.__name__}(criterion={criterion.criterion}, "
                f"smoothing={criterion.smoothing})")
    return criterion


class XentLoss(nn.Module):
    """Cross-Entropy Loss with optional label smoothing
    """
    def __init__(self, pad_index, smoothing):
        super().__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        # custom label-smoothed loss, computed with KL divergence loss
        self.criterion = nn.KLDivLoss(reduction="sum")

    def _smooth_targets(self, targets, vocab_size):
        # 进行 smoothing
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0 - self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index,
                                          as_tuple=False)
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    def _reshape(self, log_probs, targets):
        vocab_size = log_probs.size(-1)

        # reshape log_probs to (batch*seq_len x vocab_size)
        log_probs_flat = log_probs.contiguous().view(-1, vocab_size)

        targets_flat = self._smooth_targets(targets=targets.contiguous().view(-1),
                                            vocab_size=vocab_size)
        return log_probs_flat, targets_flat

    def forward(self, log_probs, trg):
        """Compute the cross-entropy between logits and targets.
        """
        log_probs, targets = self._reshape(log_probs, trg)

        # compute loss
        logits = self.criterion(log_probs, targets)
        return logits
