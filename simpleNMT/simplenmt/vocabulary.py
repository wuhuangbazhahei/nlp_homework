import logging
import sys
from collections import Counter
from pathlib import Path

from simplenmt.constants import (
    BOS_ID,
    BOS_TOKEN,
    EOS_ID,
    EOS_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    UNK_ID,
    UNK_TOKEN,
)
from simplenmt.helpers import read_list_from_file, write_list_to_file

logger = logging.getLogger(__name__)


class Vocabulary:
    """词典保存了 token 和 indices 之间的映射"""

    def __init__(self, tokens):
        """创建词典, 若 speical token 不在词典中,将其加入
        """

        self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

        self._stoi = {}  # string to index
        self._itos = []  # index to string

        # construct
        self.add_tokens(tokens=self.specials + tokens)
        assert len(self._stoi) == len(self._itos)

        # assign after stoi is built
        self.pad_index = self.lookup(PAD_TOKEN)
        self.bos_index = self.lookup(BOS_TOKEN)
        self.eos_index = self.lookup(EOS_TOKEN)
        self.unk_index = self.lookup(UNK_TOKEN)
        assert self.pad_index == PAD_ID
        assert self.bos_index == BOS_ID
        assert self.eos_index == EOS_ID
        assert self.unk_index == UNK_ID
        assert self._itos[UNK_ID] == UNK_TOKEN

    def add_tokens(self, tokens):
        """向词典中加入 tokens list
        """
        for t in tokens:
            new_index = len(self._itos)
            # add to vocab if not already there
            if t not in self._itos:
                self._itos.append(t)
                self._stoi[t] = new_index

    def to_file(self, file):
        """将词典保存为文件形式
        """
        write_list_to_file(file, self._itos)

    def is_unk(self, token: str) -> bool:
        """判断是否是 UNK
        """
        return self.lookup(token) == UNK_ID

    def lookup(self, token: str) -> int:
        """查找 token 在词典中的索引
        """
        return self._stoi.get(token, UNK_ID)

    def __len__(self) -> int:
        return len(self._itos)

    def __eq__(self, other) -> bool:
        if isinstance(other, Vocabulary):
            return self._itos == other._itos
        return False

    def array_to_sentence(self,
                          array,
                          cut_at_eos=True,
                          skip_pad=True):
        """
        Converts an array of IDs to a sentence, optionally cutting the result off at the
        end-of-sequence token.

        :param array: 1D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :param skip_pad: skip generated <pad> tokens
        :return: list of strings (tokens)
        """
        sentence = []
        for i in array:
            s = self._itos[i]
            if skip_pad and s == PAD_TOKEN:
                continue
            sentence.append(s)
            # break at the position AFTER eos
            if cut_at_eos and s == EOS_TOKEN:
                break
        return sentence

    def arrays_to_sentences(self,
                            arrays,
                            cut_at_eos,
                            skip_pad):
        """
        Convert multiple arrays containing sequences of token IDs to their sentences,
        optionally cutting them off at the end-of-sequence token.

        :param arrays: 2D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :param skip_pad: skip generated <pad> tokens
        :return: list of list of strings (tokens)
        """
        return [
            self.array_to_sentence(array=array,
                                   cut_at_eos=cut_at_eos,
                                   skip_pad=skip_pad) for array in arrays
        ]

    def sentences_to_ids(self,
                         sentences,
                         bos,
                         eos):
        """
        对当前batch中的数据进行pad
        """
        # 当前 batch 最大的长度
        max_len = max([len(sent) for sent in sentences])
        if bos:
            max_len += 1
        if eos:
            max_len += 1
        padded, lengths = [], []
        for sent in sentences:
            encoded = [self.lookup(s) for s in sent]
            if bos:
                encoded = [self.bos_index] + encoded
            if eos:
                encoded = encoded + [self.eos_index]
            offset = max(0, max_len - len(encoded))
            padded.append(encoded + [self.pad_index] * offset)
            lengths.append(len(encoded))
        return padded, lengths

    def log_vocab(self, k: int) -> str:
        """first k vocab entities"""
        return " ".join(f"({i}) {t}" for i, t in enumerate(self._itos[:k]))

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(len={self.__len__()}, "
                f"specials={self.specials})")


def sort_and_cut(counter,
                 max_size=sys.maxsize,
                 min_freq=-1):

    # 把toekn：频率 存为字典
    if min_freq > -1:
        counter = Counter({t: c for t, c in counter.items() if c >= min_freq})

    # 先按字母排序，同一字母按照词频排序，保证排序的稳定性
    tokens_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    # cut off
    vocab_tokens = [i[0] for i in tokens_and_frequencies[:max_size]]
    assert len(vocab_tokens) <= max_size, (len(vocab_tokens), max_size)
    return vocab_tokens


def _build_vocab(cfg):
    """根据 vocab_file 建立词典
    """
    
    vocab_file = cfg.get("voc_file")
    unique_tokens = read_list_from_file(Path(vocab_file))
    vocab = Vocabulary(unique_tokens)

    return vocab


def build_vocab(cfg):
    """建立词典
    """
    src_vocab = _build_vocab(cfg["src"])
    trg_vocab = _build_vocab(cfg["trg"])

    return src_vocab, trg_vocab
