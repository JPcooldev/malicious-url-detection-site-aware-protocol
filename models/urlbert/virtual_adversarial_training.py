# This file is modified from https://github.com/Davidup1/URLBERT
# Original code copyright (c) Davidup1, licensed under Apache License 2.0.

import torch
from torch import nn
import torch.nn.functional as F

def kl(inputs: torch.Tensor, targets: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    """
    KL divergence loss
    inputs: tensor, logits
    targets: tensor, logits
    """
    loss = F.kl_div(F.log_softmax(inputs, dim=-1),
                    F.softmax(targets, dim=-1),
                    reduction=reduction)
    return loss


def adv_project(grad: torch.Tensor, norm_type: str = 'inf', eps: float = 1e-6) -> torch.Tensor:
    """
    L0,L1,L2
    """
    if norm_type == 'l2':
        direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
    elif norm_type == 'l1':
        direction = grad.sign()
    else:
        direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
    return direction


def vat(model: nn.Module, 
        embeddings: torch.Tensor, 
        token_type_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        logits: torch.Tensor, 
        position_ids: torch.Tensor) -> torch.Tensor | None:
    """
    VAT
    model: nn.Module
    """
    embeddings = embeddings.detach()
    # generate noise
    noise = embeddings.new_empty(embeddings.size()).normal_(0, 1e-5).requires_grad_()

    # temporarily turn off grad tracking for model
    flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    
    # add noise to embeddings
    new_embed = embeddings + noise
    adv_output = model(inputs_embeds=new_embed,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_mask,
                       position_ids=position_ids)
    adv_logits = adv_output.logits

    # compute adversarial loss
    adv_loss = kl(adv_logits, logits.detach(), reduction="batchmean")
    delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
    norm = delta_grad.norm()


    if torch.isnan(norm) or torch.isinf(norm):
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
        return None

    # update noise
    noise = noise + delta_grad * 1e-3
    noise = adv_project(noise, norm_type='l2', eps=1e-6)
    new_embed = embeddings + noise
    new_embed = new_embed.detach()

    # turn on grad tracking for model
    for p, flag in zip(model.parameters(), flags):
        p.requires_grad_(flag)

    # compute adversarial loss
    adv_output = model(inputs_embeds=new_embed,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_mask,
                       position_ids=position_ids)
    adv_logits = adv_output.logits
    adv_loss_f = kl(adv_logits, logits.detach())
    adv_loss_b = kl(logits, adv_logits.detach())

    adv_loss = (adv_loss_f + adv_loss_b) * 1

    return adv_loss