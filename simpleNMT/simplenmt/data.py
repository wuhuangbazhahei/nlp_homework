import logging
from functools import partial

import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)

from simplenmt.batch import Batch

from simplenmt.constants import PAD_ID
from simplenmt.datasets import TranslationDataset
from simplenmt.vocabulary import build_vocab
from simplenmt.helpers import log_data_info

logger = logging.getLogger(__name__)

CPU_DEVICE = torch.device("cpu")


def load_data(data_cfg):
    """ 加载数据
    """
    src_cfg = data_cfg["src"]
    trg_cfg = data_cfg["trg"]

    # 获取 lang & path
    src_lang = src_cfg["lang"]
    trg_lang = trg_cfg["lang"]
    train_path = data_cfg.get("train", None)
    dev_path = data_cfg.get("dev", None)
    test_path = data_cfg.get("test", None)
    
    # train data
    train_data = None
    logger.info("Load train data...")
    train_data = TranslationDataset(
        path=train_path,
        src_lang=src_lang,
        trg_lang=trg_lang,
        split="train",
        has_trg=True
    )

    # 建立词典
    logger.info("Building vocabulary...")
    src_vocab, trg_vocab = build_vocab(data_cfg)

    # encoding func
    sequence_encoder = {
        src_lang: partial(src_vocab.sentences_to_ids, bos=False, eos=True),
        trg_lang: partial(trg_vocab.sentences_to_ids, bos=True, eos=True),
    }

    train_data.sequence_encoder = sequence_encoder

    # dev data
    dev_data = None
    logger.info("Loading dev set...")
    dev_data = TranslationDataset(
        path=dev_path,
        src_lang=src_lang,
        trg_lang=trg_lang,
        split="dev",
        has_trg=True,
    )
        
    dev_data.sequence_encoder = sequence_encoder
    
    test_data = None
    if test_path is not None:
        logger.info("Loading test set...")
        test_data = TranslationDataset(
            path=test_path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split="test",
            has_trg=True
        )
        test_data.sequence_encoder = sequence_encoder
    
    logger.info("Data loaded.")
    if test_data is None:
        log_data_info(src_vocab, trg_vocab, train_data, dev_data)
        return src_vocab, trg_vocab, train_data, dev_data
    else:
        return src_vocab, trg_vocab, train_data, dev_data, test_data

def collate_fn(
    batch,
    src_process,
    trg_process,
    pad_index=PAD_ID,
    device=CPU_DEVICE,
    has_trg=True,
    is_train=True,
):
    """一个 mini-batch 的组织方式
    """
    batch = [(s, t) for s, t in batch]
    src_list, trg_list = zip(*batch)

    src, src_length = src_process(src_list)

    if has_trg:
        trg, trg_length = trg_process(trg_list)
    else:
        trg, trg_length = None, None
    return Batch(
        src=torch.tensor(src).long(),
        src_length=torch.tensor(src_length).long(),
        trg=torch.tensor(trg).long() if trg else None,
        trg_length=torch.tensor(trg_length).long() if trg_length else None,
        device=device,
        pad_index=pad_index,
        has_trg=has_trg,
        is_train=is_train,
    )


def make_data_iter(
    dataset,
    batch_size,
    seed=42,
    shuffle=False,
    pad_index=PAD_ID,
    device=CPU_DEVICE,
):
    """返回 DataLoader
    """
    if shuffle and dataset.split == "train":
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)

    batch_sampler = TokenBatchSampler(sampler, batch_size=batch_size, drop_last=False)

    # data iterator
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=partial(
            collate_fn,
            src_process=dataset.sequence_encoder[dataset.src_lang],
            trg_process=dataset.sequence_encoder[dataset.trg_lang],
            pad_index=pad_index,
            device=device,
            has_trg=dataset.has_trg,
            is_train=dataset.split == "train",
        )
    )
    
    
class TokenBatchSampler(BatchSampler):
    """按 token num 来组织 mini-batch
    """

    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        batch = []
        max_tokens = 0
        d = self.sampler.data_source
        for idx in self.sampler:
            src, trg = d[idx]  # call __getitem__()
            if src is not None:  # otherwise drop instance
                src_len = 0 if src is None else len(src)
                trg_len = 0 if trg is None else len(trg)
                n_tokens = 0 if src_len == 0 else max(src_len + 1, trg_len + 2)
                batch.append(idx)
                if n_tokens > max_tokens:
                    max_tokens = n_tokens
                if max_tokens * len(batch) >= self.batch_size:
                    yield batch
                    batch = []
                    max_tokens = 0
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        raise NotImplementedError