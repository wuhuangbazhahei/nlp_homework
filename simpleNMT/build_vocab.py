import argparse
import sys
from collections import Counter
from pathlib import Path

from simplenmt.helpers import load_config, write_list_to_file, flatten
from simplenmt.vocabulary import sort_and_cut
from simplenmt.datasets import TranslationDataset

def build_vocab_from_sents(
    tokens,
    min_freq,
    vocab_file,
    max_size,
):
    """
    从英文、德文合并的text集中读取
    """
    print("### Building vocab...")
    # newly create unique token list
    counter = Counter(flatten(tokens))#从大到小
    unique_tokens = sort_and_cut(counter, max_size, min_freq)
    write_list_to_file(vocab_file, unique_tokens)


def run(
    args,
    train_data,
    langs,
    min_freq,
    max_size,
    vocab_file
):
    if vocab_file.is_file():
        print(f"### Vocab file {vocab_file} will be overwritten.")
    
    def _get_sents(args, dataset, langs):
        assert len(langs) in [1, 2], langs

        sents = []
        for lang in langs:
            sents.extend(dataset.get_list(lang=lang))
        return sents
    
    sents = _get_sents(args, train_data, langs)
    tokens = [sent.split(' ') for sent in sents]

    build_vocab_from_sents(
        tokens=tokens,
        max_size=max_size,
        min_freq=min_freq,
        vocab_file=vocab_file,
    )
    
def main(args):
    cfg = load_config(Path(args.config_path))
    src_cfg = cfg["data"]["src"]
    trg_cfg = cfg["data"]["trg"]
    
    train_data = TranslationDataset(
        path=cfg["data"]["train"],
        src_lang=src_cfg["lang"],
        trg_lang=trg_cfg["lang"],
        split="train",
    ) #返回一个字典train_data{src：源语言句子list，tgt：目标语言句子list}
    def _parse_cfg(cfg):
        lang = cfg["lang"]
        level = cfg["level"]
        min_freq = cfg.get("voc_min_freq", 1)
        max_size = int(cfg.get("voc_limit", sys.maxsize))
        voc_file = Path(cfg.get("voc_file", "vocab.txt"))
        tok_type = cfg.get("tokenizer_type", "sentencepiece")
        tok_cfg = cfg.get("tokenizer_cfg", {})
        return lang, level, min_freq, max_size, voc_file, tok_type, tok_cfg

    src_tuple = _parse_cfg(src_cfg)#读取源语言参数
    trg_tuple = _parse_cfg(trg_cfg)

    if args.joint:
        for s, t in zip(src_tuple[1:], trg_tuple[1:]):
            assert s == t
        
        run(args,
            train_data=train_data,
            langs=[src_tuple[0], trg_tuple[0]],
            min_freq=src_tuple[2],
            max_size=src_tuple[3],
            vocab_file=src_tuple[4],
        )
    
    
    
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="build joint vocab")
    
    ap.add_argument("config_path", type=str)
    ap.add_argument("--joint", action="store_true")
    args = ap.parse_args()
    main(args)