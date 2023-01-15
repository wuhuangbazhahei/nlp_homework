from pathlib import Path
import sys
import yaml
import shutil
import logging
import functools
import operator
import re
import torch
import numpy as np
import random

def load_config(path):
    """path: yaml 文件路径 加载yaml中的参数
    """
    if isinstance(path, str):
        path = Path(path)
        
    with path.open("r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    return cfg


def flatten(array):
    """将 2D list 展开，效率更高
    """
    return functools.reduce(operator.iconcat, array, [])


def set_seed(seed: int) -> None:
    """设置随机种子
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
        
        
def make_logger(log_dir, mode="train"):
    """Create a logger for logging the training/testing process.
    """
    logger = logging.getLogger("")  # root logger

    # add handlers only once.
    if len(logger.handlers) == 0:
        logger.setLevel(level=logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s")

        if log_dir is not None:
            if log_dir.is_dir():
                log_file = log_dir / f"{mode}.log"

                fh = logging.FileHandler(log_file.as_posix())
                fh.setLevel(level=logging.DEBUG)
                logger.addHandler(fh)
                fh.setFormatter(formatter)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)

        logger.addHandler(sh)


def log_cfg(cfg, prefix="cfg") -> None:
    """Write configuration to log.
    """
    logger = logging.getLogger(__name__)
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = ".".join([prefix, k])
            log_cfg(v, prefix=p)
        else:
            p = ".".join([prefix, k])
            logger.info("%34s : %s", p, v)


def make_model_dir(model_dir, overwrite):
    """Create a new directory for the model.
    """
    model_dir = model_dir.absolute()
    if model_dir.is_dir():
        if not overwrite:
            raise FileExistsError(f"Model directory {model_dir} exists "
                                  f"and overwriting is disabled.")
        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)  # create model_dir recursively
    return model_dir


def read_list_from_file(input_path):
    """ 读取bpe分词之后的文件存成list
    """
    if input_path is None:
        return []

    return [
        line.rstrip("\n")
        for line in input_path.read_text(encoding="utf-8").splitlines()
    ]
    

def write_list_to_file(output_path, array):
    """把list数据写入文件
    """
    with output_path.open("w", encoding="utf-8") as opened_file:
        for entry in array:
            opened_file.write(f"{entry}\n")
            

def log_data_info(
    src_vocab,
    trg_vocab,
    train_data,
    valid_data,
):
    """打印部分 data 和 vocabulary
    """
    logger = logging.getLogger(__name__)
    logger.info("Train dataset: %s", train_data)
    logger.info("Valid dataset: %s", valid_data)

    if train_data:
        src = "\n\t[SRC] " + " ".join(
            train_data.get_item(index=0, lang=train_data.src_lang))
        trg = "\n\t[TRG] " + " ".join(
            train_data.get_item(index=0, lang=train_data.trg_lang))
        logger.info("First training example:%s%s", src, trg)

    logger.info("First 10 Src tokens: %s", src_vocab.log_vocab(10))
    logger.info("First 10 Trg tokens: %s", trg_vocab.log_vocab(10))

    logger.info("Number of unique Src tokens (vocab_size): %d", len(src_vocab))
    logger.info("Number of unique Trg tokens (vocab_size): %d", len(trg_vocab))
    
def parse_train_args(cfg):
    """Parse and validate train args specified in config file"""

    model_dir = Path(cfg["model_dir"])
    patience = cfg.get("patience")
    label_smoothing = cfg.get("label_smoothing", 0.01)
    use_cuda = cfg["use_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # save/delete checkpoints
    keep_best_ckpts = int(cfg.get("keep_best_ckpts", 5))

    # logging, validation
    logging_freq = cfg.get("logging_freq", 100)

    # data & batch handling
    seed = cfg.get("random_seed", 42)
    shuffle = cfg.get("shuffle", True)
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]

    return (
        model_dir,
        patience,
        label_smoothing,
        keep_best_ckpts,
        logging_freq,
        seed,
        shuffle,
        epochs,
        batch_size,
        device
    )


def parse_model_args(cfg):
    """Parse and validate train args specified in config file"""
    
    d_model = cfg.get("d_model", 512)
    d_ff = cfg.get("d_ff", 2048)
    num_layers = cfg.get("num_layers", 6)
    num_heads = cfg.get("num_heads", 8)
    

    return (
        d_model,
        d_ff,
        num_layers,
        num_heads
    )
    
def parse_test_args(cfg):
    """Parse test args"""
    # batch options
    batch_size = cfg.get("batch_size", 64)

    # limit on generation length
    max_output_length = cfg.get("max_output_length", -1)
    min_output_length = cfg.get("min_output_length", 1)

    # eval metrics
    eval_metrics = [s.strip().lower() for s in cfg["eval_metrics"]]

    # sacrebleu cfg
    sacrebleu_cfg = cfg.get("sacrebleu_cfg", {})

    # beam search options
    n_best = cfg.get("n_best", 1)
    beam_size = cfg.get("beam_size", 1)
    beam_alpha = cfg.get("beam_alpha", -1)

    # control options
    generate_unk = cfg.get("generate_unk", True)

    return (
        batch_size,
        max_output_length,
        min_output_length,
        eval_metrics,
        sacrebleu_cfg,
        beam_size,
        beam_alpha,
        n_best,
        generate_unk
    )

def remove_extra_spaces(s: str) -> str:
    """
    Remove extra spaces
    - used in pre_process() / post_process() in tokenizer.py
    :param s: input string
    :return: string w/o extra white spaces
    """
    s = re.sub("\u200b", "", s)
    s = re.sub("[ 　]+", " ", s)

    s = s.replace(" ?", "?")
    s = s.replace(" !", "!")
    s = s.replace(" ,", ",")
    s = s.replace(" .", ".")
    s = s.replace(" :", ":")
    return s.strip()

def load_checkpoint(path, device):
    """
    Load model from saved checkpoint.
    """
    logger = logging.getLogger(__name__)
    model_checkpoint = torch.load(path, map_location=device)
    logger.info("Load model from %s.", path)
    
    return model_checkpoint