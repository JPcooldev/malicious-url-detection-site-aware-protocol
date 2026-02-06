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


# ContrastiveLoss Define
def calculateContrastiveLoss(emb_i: torch.Tensor, 
                             emb_j: torch.Tensor, 
                             batch_size: int, 
                             temperature: float, 
                             device: torch.device) -> torch.Tensor:
    z_i = F.normalize(emb_i, dim=1)
    z_j = F.normalize(emb_j, dim=1)

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    sim_ij = torch.diag(similarity_matrix, batch_size)
    sim_ji = torch.diag(similarity_matrix, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    nominator = torch.exp(positives / temperature)
    negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(device)
    denominator = negatives_mask * torch.exp(similarity_matrix / temperature)

    loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
    loss = torch.sum(loss_partial) / (2 * batch_size)
    return loss


# Generate loss
def sscl(model: nn.Module, 
         input_ids: torch.Tensor, 
         token_types: torch.Tensor, 
         atten_masks: torch.Tensor, 
         position_ids: torch.Tensor, 
         device: torch.device) -> torch.Tensor | None:
    # obtain input embeddings and true logits 
    # (no need for grad tracking and no_grad will still enable dropout to produce two different outputs)
    with torch.no_grad():
        out1 = model(input_ids,
                       token_type_ids=token_types,
                       attention_mask=atten_masks,
                       position_ids=position_ids,
                       output_hidden_states=True,
                       )
        out2 = model(input_ids,
                       token_type_ids=token_types,
                       attention_mask=atten_masks,
                       position_ids=position_ids,
                       output_hidden_states=True,
                       )
        embeddings1 = out1.hidden_states[0].detach()
        embeddings2 = out2.hidden_states[0].detach()
        logits1 = out1.logits.detach()
        logits2 = out2.logits.detach()

    # generate noise
    noise1 = embeddings1.new_empty(embeddings1.size()).normal_(0, 1e-5).requires_grad_()
    noise2 = embeddings2.new_empty(embeddings2.size()).normal_(0, 1e-5).requires_grad_()

    # temporarily turn off grad tracking for model
    flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    # Fast Gradient Sign Method (FGSM)
    # for two iterations alter noise using gradient of loss with respect to noise
    epochs = 2
    for _ in range(epochs):
        # add noise to embeddings
        new_embeddings1 = embeddings1 + noise1
        new_embeddings2 = embeddings2 + noise2

        adv_output1 = model(inputs_embeds=new_embeddings1,
                            token_type_ids=token_types,
                            attention_mask=atten_masks,
                            position_ids=position_ids,
                            )
        adv_output2 = model(inputs_embeds=new_embeddings2,
                            token_type_ids=token_types,
                            attention_mask=atten_masks,
                            position_ids=position_ids,
                            )
        adv_logits1 = adv_output1.logits
        adv_logits2 = adv_output2.logits

        adv_loss1 = kl(adv_logits1, logits1, reduction="batchmean")
        adv_loss2 = kl(adv_logits2, logits2, reduction="batchmean")

        # compute gradient of loss with respect to noise (adversarial perturbation)
        delta_grad1, = torch.autograd.grad(adv_loss1, noise1, only_inputs=True)
        delta_grad2, = torch.autograd.grad(adv_loss2, noise2, only_inputs=True)
        norm1 = delta_grad1.norm()
        norm2 = delta_grad2.norm()

        if torch.isnan(norm1) or torch.isinf(norm1) or torch.isnan(norm2) or torch.isinf(norm2):
            for p, flag in zip(model.parameters(), flags):
                p.requires_grad_(flag)
            return None

        # update noise
        noise1 = noise1 + delta_grad1 * 1e-1
        noise2 = noise2 + delta_grad2 * 1e-5

        # normalize noise
        noise1 = adv_project(noise1, norm_type='l2', eps=1e-6)
        noise2 = adv_project(noise2, norm_type="l2", eps=1e-6)

    # add final noise to the embeddings (detach from the current graph)
    new_embeddings1 = embeddings1 + noise1
    new_embeddings1 = new_embeddings1.detach()
    new_embeddings2 = embeddings2 + noise2
    new_embeddings2 = new_embeddings2.detach()

    # turn on grad tracking for model
    for p, flag in zip(model.parameters(), flags):
        p.requires_grad_(flag)

    adv_output1 = model(inputs_embeds=new_embeddings1,
                        token_type_ids=token_types,
                        attention_mask=atten_masks,
                        position_ids=position_ids,
                        output_hidden_states=True,
                        )
    adv_output2 = model(inputs_embeds=new_embeddings2,
                        token_type_ids=token_types,
                        attention_mask=atten_masks,
                        position_ids=position_ids,
                        output_hidden_states=True,
                        )

    # obtain adversarial output embeddings for [CLS] token
    adv_hidden_states1 = adv_output1.hidden_states[-1][:, 0, :]
    adv_hidden_states2 = adv_output2.hidden_states[-1][:, 0, :]
    batch_size = input_ids.size(0)
    loss = calculateContrastiveLoss(adv_hidden_states1, adv_hidden_states2, batch_size, 0.1, device)
    return loss