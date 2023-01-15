import logging
import math
import time

from pathlib import Path

import torch
from tqdm import tqdm
import torch.nn.functional as F
from simplenmt.data import load_data, make_data_iter
from simplenmt.search import greedy_search, beam_search
from simplenmt.constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, PAD_ID, BOS_ID, EOS_ID
from simplenmt.data import make_data_iter
from simplenmt.transformer import Transformer
from simplenmt.helpers import(
    load_checkpoint,
    parse_model_args,
    load_config,
    set_seed
)

from simplenmt.loss import calculate_loss

logger = logging.getLogger(__name__)


def predict(
    model,
    data,
    device,
    cfg=None,
    criterion=None
):

    logger.info("Validing...")
    eval_batch_size = cfg.get("batch_size")
    valid_iter = make_data_iter(
        dataset=data,
        batch_size=eval_batch_size,
        shuffle=False,
        pad_index=PAD_ID,
        device=device,
    )

    model.eval()

    valid_scores = {"loss": float("nan"), "acc": float("nan"), "ppl": float("nan")}
    total_loss = 0
    total_nseqs = 0
    total_ntokens = 0
    total_n_correct = 0

    for batch in valid_iter:
        total_nseqs += batch.nseqs  # number of sentences in the current batch
    
        with torch.no_grad():
            src_tokens, trg_input, trg, trg_mask = batch.src, batch.trg_input, batch.trg, batch.trg_mask
            model_out = model(src_tokens, trg_input)
            log_probs = F.log_softmax(model_out, dim=-1)
            batch_loss, log_probs, n_correct = calculate_loss(criterion=criterion, log_probs=log_probs,
                                                                trg=trg, trg_mask=trg_mask)

        total_loss += batch_loss.item()  # cast Tensor to float
        total_n_correct += n_correct.item()  # cast Tensor to int
        total_ntokens += batch.ntokens
    
    # 计算 loss
    # normalized loss
    valid_scores["loss"] = total_loss / total_ntokens
    # accuracy before decoding
    valid_scores["acc"] = total_n_correct / total_ntokens
    # exponent of token-level negative log likelihood
    valid_scores["ppl"] = math.exp(total_loss / total_ntokens)

    return valid_scores

def generate(
    model,
    data,
    device,
    cfg=None,
    path=None,
    beam_size=0,
    length_penalty=0,
    max_seq_len=0,
    src_vocab=None,
    trg_vocab=None
):

    logger.info("generating...")
    test_batch_size = cfg.get("batch_size")
    test_iter = make_data_iter(
        dataset=data,
        batch_size=test_batch_size,
        shuffle=False,
        pad_index=PAD_ID,
        device=device,
    )
    result_path = path + '/result.txt'
    start_time = time.time()
    with open(result_path, 'w', encoding='utf8') as f, torch.no_grad(): 
        for batch in test_iter:
            src, trg = batch.src, batch.trg
            if beam_size > 0:
                pred_tokens, _ = beam_search(model=model, src_tokens=src,
                        beam_size=beam_size, length_penalty=length_penalty,
                        max_seq_len=max_seq_len, bos=BOS_ID,
                        eos=EOS_ID, src_pdx=PAD_ID, tgt_pdx=PAD_ID)
            else:
                pred_tokens = greedy_search(model=model, src_tokens=src,
                        max_seq_len=max_seq_len, bos=BOS_ID,
                        eos=EOS_ID, src_pdx=PAD_ID, tgt_pdx=PAD_ID)

            def de_numericalize(vocab, tokens):
                remove_constants={PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}

                sentences = list()
                for row in tokens:
                    words_list = list()
                    for word_id in row:
                        word = vocab._itos[word_id]
                        if word == EOS_TOKEN:
                            break
                        if word not in remove_constants:
                            words_list.append(word)
                    sentences.append(words_list)

                return sentences

            src_sentences = de_numericalize(src_vocab, src)
            tgt_sentences = de_numericalize(trg_vocab, trg)
            pred_sentences = de_numericalize(trg_vocab, pred_tokens)

            for src_words, tgt_words, pred_words in zip(src_sentences, tgt_sentences, pred_sentences):
                content = '-S\t{}\n-T\t{}\n-P\t{}\n\n'.format(
                    ' '.join(src_words), ' '.join(tgt_words), ' '.join(pred_words))            
                f.write(content)

    print('Successful. Generate time: {:.1f} min, the result has saved at {}'
            .format((time.time() - start_time) / 60, result_path))



def test(
    cfg_file,
    ckpt,
    output_path=None,
) -> None:
    """解码
    """
    cfg = load_config(Path(cfg_file))
    (
        d_model,
        d_ff,
        num_layers,
        num_heads
    ) = parse_model_args(cfg["model"])
    
    src_vocab, trg_vocab, _, _, test_data = load_data(data_cfg=cfg["data"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = Transformer(len(src_vocab), len(trg_vocab), src_pdx=PAD_ID, tgt_pdx=PAD_ID,
                        d_model=d_model, d_ff=d_ff, n_head=num_heads, 
                        n_encoder_layers=num_layers, n_decoder_layers=num_layers)
    # load model checkpoint
    model_checkpoint = load_checkpoint(ckpt, device=device)

    # restore model and optimizer parameters
    model.load_state_dict(model_checkpoint["model"])
    if device.type == "cuda":
        model.to(device)

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))
    
    generate(
        model=model,
        data=test_data,
        device=device,
        cfg=cfg["testing"],
        path=output_path,
        beam_size=5,
        length_penalty=0.7,
        max_seq_len=128,
        src_vocab=src_vocab,
        trg_vocab=trg_vocab
    )
            
