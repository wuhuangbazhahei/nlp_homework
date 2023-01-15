import torch

def calculate_loss(criterion, log_probs, trg, trg_mask):
        batch_loss = criterion(log_probs, trg)
        trg_mask = trg_mask.squeeze(1)
        n_correct = torch.sum(
            log_probs.argmax(-1).masked_select(trg_mask).eq(trg.masked_select(trg_mask)))

        return_tuple = (batch_loss, log_probs, n_correct)
        return tuple(return_tuple)
    

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def vanilla_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
    return nll_loss

    
def compute_kl_loss(model, net_output, pad_mask=None, reduce=True, reg_alpha=0.3):
    net_prob = model.get_normalized_probs(net_output, log_probs=True)
    net_prob_tec = model.get_normalized_probs(net_output, log_probs=False)

    p, q = torch.split(net_prob, net_prob.size(0)//2, dim=0)
    p_tec, q_tec = torch.split(net_prob_tec, net_prob_tec.size(0)//2, dim=0)
    
    p_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none')
    q_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none')
    
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    if reduce:
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

    kl_loss = (p_loss + q_loss) / 2
    loss = reg_alpha * kl_loss
    return loss
