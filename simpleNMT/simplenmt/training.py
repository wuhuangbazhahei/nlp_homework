from pathlib import Path
import shutil
import os
import time
import math
import heapq
import torch
import torch.nn.functional as F
import logging

from simplenmt.data import load_data, make_data_iter
from simplenmt.model import Transformer
from simplenmt.builder import build_optimizer, build_criterion
from simplenmt.prediction import predict
from simplenmt.constants import PAD_ID
from simplenmt.loss import calculate_loss
from simplenmt.batch import Batch
from simplenmt.helpers import (
    load_config, 
    make_model_dir, 
    log_cfg, 
    make_logger, 
    set_seed, 
    parse_train_args, 
    parse_model_args
    )

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, cfg, model):
        
        (
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
        ) = parse_train_args(cfg["training"])
        
        # train
        self.patience = patience
        self.label_smoothing = label_smoothing
        self.logging_freq = logging_freq

        # model
        self.model_dir = model_dir
        self.model = model
        
        self.device = device
        if self.device.type == "cuda":
            self.model.to(self.device)
        
        # optimizer & criterion
        self.optimizer = build_optimizer(config=cfg["training"],
                                         parameters=self.model.parameters())
        self.criterion = build_criterion(label_smoothing=self.label_smoothing, pad_index=PAD_ID)
        
        # 保存 ckpt 数量
        self.num_ckpts = keep_best_ckpts
        self.ckpt_queue = []

        # data 处理有关
        self.seed = seed
        self.shuffle = shuffle
        self.epochs = epochs
        self.batch_size = batch_size

        # 一些静态变量
        self.total_tokens = 0
        self.steps = 0
        self.best_ckpt_iter = 0
        self.best_ckpt_score = 1e9
        # valid
        self.valid_cfg = cfg["testing"].copy()
        self.valid_cfg["batch_size"] = self.batch_size
        self.early_stopping_metric = "ppl"
        
    def train_and_validate(self, train_data, valid_data):
        """训练 & 验证
        """
        self.train_iter = make_data_iter(
            dataset=train_data,
            batch_size=self.batch_size,
            seed=self.seed,
            shuffle=self.shuffle,
            device=self.device.type,
            pad_index=PAD_ID,
        )

        logger.info(
            "Train stats:\n"
            "\tdevice: %s\n"
            "\tbatch size per device: %d\n",
            self.device.type,  # next(self.model.parameters()).device
            self.batch_size
        )

        for epoch_no in range(self.epochs):
            logger.info("EPOCH %d", epoch_no + 1)
            
            self.model.train()

            # Reset statistics for each epoch.
            start = time.time()
            total_valid_duration = 0
            start_tokens = self.total_tokens
            self.model.zero_grad()
            epoch_loss = 0
            total_batch_loss = 0
            total_n_correct = 0

            batch: Batch
            for _, batch in enumerate(self.train_iter):
                # sort batch now by src length and keep track of order
                batch.sort_by_src_length()

                # get batch loss
                norm_batch_loss, n_correct = self._train_step(batch)
                total_batch_loss += norm_batch_loss
                total_n_correct += n_correct

                # make gradient step
                self.optimizer.step()
                self.model.zero_grad()
                self.steps += 1

                # log learning progress
                if self.steps % self.logging_freq == 0:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_tok = self.total_tokens - start_tokens
                    token_accuracy = total_n_correct / elapsed_tok

                    logger.info(
                        "Epoch %3d, "
                        "Step: %8d, "
                        "Batch Loss: %12.6f, "
                        "Batch Acc: %.6f, "
                        "Tokens per Sec: %8.0f, "
                        "Lr: %.6f",
                        epoch_no + 1,
                        self.steps,
                        total_batch_loss,
                        token_accuracy,
                        elapsed_tok / elapsed,
                        self.optimizer.param_groups[0]["lr"],
                    )
                    start = time.time()
                    total_valid_duration = 0
                    start_tokens = self.total_tokens

                # update epoch_loss
                epoch_loss += total_batch_loss  # accumulate loss
                total_batch_loss = 0  # reset batch loss
                total_n_correct = 0  # reset batch accuracy


            valid_duration = self._validate(epoch_no ,valid_data)
            total_valid_duration += valid_duration

            logger.info(
                "Epoch %3d: total training loss %.2f",
                epoch_no + 1,
                epoch_loss,
            )

        logger.info(
            "Best validation result at step %8d: %6.2f.",
            self.best_ckpt_iter,
            self.best_ckpt_score,
        )
        
    def _train_step(self, batch):
        """
        Train the model on one batch: Compute the loss.
        :param batch: training batch
        :return:
            - losses for batch (sum)
            - number of correct tokens for batch (sum)
        """
        self.model.train()
        src_tokens, trg_input, trg, trg_mask = batch.src, batch.trg_input, batch.trg, batch.trg_mask
        model_out = self.model(src_tokens, trg_input)
        
        log_probs = F.log_softmax(model_out, dim=-1)
        batch_loss, _, correct_tokens = calculate_loss(self.criterion, log_probs=log_probs, trg=trg, trg_mask=trg_mask)

        norm_batch_loss = batch_loss / batch.ntokens

        norm_batch_loss.backward()

        # increment token counter
        self.total_tokens += batch.ntokens

        return norm_batch_loss.item(), correct_tokens.item()
    
    def _validate(self, epoch, valid_data):
        
        valid_start_time = time.time()

        valid_scores = predict(
            model=self.model,
            data=valid_data,
            device=self.device.type,
            cfg=self.valid_cfg,
            criterion=self.criterion
        )
        valid_ppl, valid_acc, valid_loss = valid_scores["ppl"], valid_scores["acc"], valid_scores["loss"]
        valid_duration = time.time() - valid_start_time
        logger.info("validation duration: {}".format(valid_duration))
        ckpt_score = valid_scores[self.early_stopping_metric]
        logger.info("validation ppl: {}, validation acc: {}, validation loss: {}".format(ckpt_score, valid_acc, valid_loss))
        # update new best
        if ckpt_score < self.best_ckpt_score:
            self.best_ckpt_score = ckpt_score
            self.best_ckpt_iter = epoch
            logger.info(
                "New best validation result [{}]={}!".format(
                self.early_stopping_metric,valid_ppl
            ))
            self._save_model()

        # append to validation report
        self._add_report(valid_scores=valid_scores, new_best=self.best_ckpt_score)

        return valid_duration

    def _save_model(self):
        """保存 ckpt
        """
        params = self.model.state_dict()

        checkpoint = {'epoch': self.best_ckpt_iter, 'model': params}
        torch.save(checkpoint, '{}/checkpoint_{}.pt'.format(self.model_dir, self.best_ckpt_iter))
        self.ckpt_queue.append(self.best_ckpt_iter)
        if len(self.ckpt_queue) > self.num_ckpts:
            ckpt_suffix = self.ckpt_queue.pop(0)
            to_del_ckpt = '{}/checkpoint_{}.pt'.format(self.model, ckpt_suffix)
            if os.path.exists(to_del_ckpt):
                os.remove(to_del_ckpt)
                          
        torch.save(checkpoint, '{}/checkpoint_best.pt'.format(self.model_dir))
    
    
    def _add_report(self, valid_scores, new_best):
        current_lr = self.optimizer.param_groups[0]["lr"]

        valid_file = self.model_dir / "validations.txt"
        with valid_file.open("a", encoding="utf-8") as opened_file:
            score_str = "\t".join([f"Steps: {self.steps}"] + [
                f"{eval_metric}: {score:.5f}"
                for eval_metric, score in valid_scores.items() if not math.isnan(score)
            ] + [f"LR: {current_lr:.8f}", "*" if new_best else ""])
            opened_file.write(f"{score_str}\n")
        
        
def train(cfg_file):
    # 加载 yaml config 文件
    cfg = load_config(Path(cfg_file))
    
    # 创建模型文件
    model_dir = make_model_dir(
        Path(cfg["training"]["model_dir"]),
        overwrite=cfg["training"].get("overwrite", True),
    )
    
    # 日志相关
    make_logger(model_dir, mode='train')
    log_cfg(cfg)
    
    # 将 config 文件复制到 model_dir 进行保存
    shutil.copy2(cfg_file, (model_dir / "config.yaml").as_posix())

    # 随机数种子
    set_seed(seed=cfg["training"].get("random_seed", 42))

    # 加载数据
    src_vocab, trg_vocab, train_data, dev_data, _ = load_data(data_cfg=cfg["data"])

    # 复制词表到 model_dir 中保存
    src_vocab.to_file(model_dir / "src_vocab.txt")
    trg_vocab.to_file(model_dir / "trg_vocab.txt")

    (
        d_model,
        d_ff,
        num_layers,
        num_heads
    ) = parse_model_args(cfg["model"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(len(src_vocab), len(trg_vocab), d_model, d_ff, num_layers, num_heads, device=device)
    
    trainer = Trainer(cfg=cfg, model=model)

    trainer.train_and_validate(train_data, dev_data)
    