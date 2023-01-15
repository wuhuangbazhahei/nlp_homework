import logging
import numpy as np
import torch
from torch import Tensor

from simplenmt.constants import PAD_ID

logger = logging.getLogger(__name__)


class Batch:
    def __init__(
        self,
        src,
        src_length,
        trg,
        trg_length,
        device=torch.device,
        pad_index= PAD_ID,
        has_trg=True,
        is_train=True,
    ):
        self.src = src
        self.src_length = src_length
        self.src_mask = (self.src != pad_index).unsqueeze(1)
        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_length = None

        self.nseqs = self.src.size(0)
        self.ntokens = None
        self.has_trg = has_trg
        self.is_train  = is_train

        if self.has_trg:
            assert trg is not None and trg_length is not None
            # trg_input is used for teacher forcing, last one is cut off
            self.trg_input = trg[:, :-1]  # shape (batch_size, seq_length) decoder端输入 不包含eos
            self.trg_length = trg_length - 1
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg[:, 1:]  # shape (batch_size, seq_length) decoder端输出 包含bos
            # we exclude the padded areas (and blank areas) from the loss computation
            self.trg_mask = (self.trg != pad_index).unsqueeze(1)
            self.ntokens = (self.trg != pad_index).data.sum().item()

        if device == "cuda":
            self._make_cuda(device)

        # a batch has to contain more than one src sentence
        assert self.nseqs > 0, self.nseqs

    def _make_cuda(self, device: torch.device) -> None:
        """Move the batch to GPU"""
        self.src = self.src.to(device)
        self.src_length = self.src_length.to(device)
        self.src_mask = self.src_mask.to(device)

        if self.has_trg:
            self.trg_input = self.trg_input.to(device)
            self.trg = self.trg.to(device)
            self.trg_mask = self.trg_mask.to(device)

    def normalize(
        self,
        tensor,
        normalization,
    ):
        """
        Normalizes batch tensor (i.e. loss). Takes sum over multiple gpus, divides by
        nseqs or ntokens
        """
        # TODO: n_gpus & 梯度累积
        if normalization == "sum":  # pylint: disable=no-else-return
            return tensor
        elif normalization == "batch":
            normalizer = self.nseqs
        elif normalization == "tokens":
            normalizer = self.ntokens
        elif normalization == "none":
            normalizer = 1

        norm_tensor = tensor / normalizer
        
        return norm_tensor

    def sort_by_src_length(self):
        """
        Sort by src length (descending) and return index to revert sort
        :return: list of indices
        """
        _, perm_index = self.src_length.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        sorted_src_length = self.src_length[perm_index]
        sorted_src = self.src[perm_index]
        sorted_src_mask = self.src_mask[perm_index]
        if self.has_trg:
            sorted_trg_input = self.trg_input[perm_index]
            sorted_trg_length = self.trg_length[perm_index]
            sorted_trg_mask = self.trg_mask[perm_index]
            sorted_trg = self.trg[perm_index]

        self.src = sorted_src
        self.src_length = sorted_src_length
        self.src_mask = sorted_src_mask

        if self.has_trg:
            self.trg_input = sorted_trg_input
            self.trg_mask = sorted_trg_mask
            self.trg_length = sorted_trg_length
            self.trg = sorted_trg

        assert max(rev_index) < len(rev_index), rev_index
        return rev_index

    def score(self, log_probs: Tensor):
        """Look up the score of the trg token (ground truth) in the batch"""
        scores = []
        for i in range(self.nseqs):
            scores.append(
                np.array([
                    log_probs[i, j, ind].item() for j, ind in enumerate(self.trg[i])
                    if ind != PAD_ID
                ]))
        # Note: each element in `scores` list can have different lengths.
        return np.array(scores, dtype=object)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nseqs={self.nseqs}, ntokens={self.ntokens}, "
            f"has_trg={self.has_trg}, is_train={self.is_train})")