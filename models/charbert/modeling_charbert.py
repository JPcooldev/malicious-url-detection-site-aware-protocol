# This file is modified from https://github.com/mawentao277/CharBERT
# Original code copyright (c) mawentao277, licensed under Apache License 2.0.

from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import (
    BertLayer,
    BertPreTrainedModel,
    BertPredictionHeadTransform,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from models.charbert.configuration_charbert import CharBertConfig


class CharBertEmbeddings(nn.Module):
    def __init__(self, config: CharBertConfig):
        super().__init__()
        self.config = config
        # Embedding for character channel
        self.char_embeddings = nn.Embedding(
            self.config.char_vocab_size,
            self.config.char_embedding_size, 
            padding_idx=self.config.pad_token_id
        )
        # RNN layer for character channel (creates a final embedding from all charachters in the token)
        self.rnn_layer = nn.GRU(
            input_size= self.config.char_embedding_size,
            hidden_size=int(self.config.hidden_size / 4), 
            batch_first=True, 
            bidirectional=True
        )

    def forward(self, 
                char_input_ids: torch.Tensor, 
                start_ids: torch.Tensor, 
                end_ids: torch.Tensor) -> torch.Tensor:
        input_shape = char_input_ids.size()
        assert len(input_shape) == 2
        
        batch_size, char_maxlen = input_shape[0], input_shape[1]
        char_input_ids_reshape = torch.reshape(char_input_ids, (batch_size, char_maxlen))
        char_embeddings = self.char_embeddings(char_input_ids_reshape)

        self.rnn_layer.flatten_parameters()
        # bidirectional GRU that processes entire character sequence (all characters in the sequence, not just in the token)
        all_hiddens, last_hidden = self.rnn_layer(char_embeddings)
         
        start_one_hot = nn.functional.one_hot(start_ids, num_classes=char_maxlen)
        end_one_hot   = nn.functional.one_hot(end_ids, num_classes=char_maxlen)
        # for each token, get the hidden state of the first and last character
        start_hidden  = torch.matmul(start_one_hot.float(), all_hiddens)
        end_hidden    = torch.matmul(end_one_hot.float(), all_hiddens)

        # concatenate the hidden states of the first and last character
        char_embeddings_repr = torch.cat([start_hidden, end_hidden], dim=-1)
        return char_embeddings_repr

class BertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config: CharBertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, 
                input_ids: Optional[torch.Tensor] = None, 
                token_type_ids: Optional[torch.Tensor] = None, 
                position_ids: Optional[torch.Tensor] = None, 
                inputs_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # sum word embeddings, position embeddings and token type embeddings
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings 
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CharBertEncoder(nn.Module):
    def __init__(self, config: CharBertConfig):
        super().__init__()
        self.output_attentions = getattr(config, "output_attentions", False)
        self.output_hidden_states = getattr(config, "output_hidden_states", False)
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        
        self.word_linear1 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.word_linear2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        
        self.char_linear1 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.char_linear2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        # fusion layer for character and word channels
        self.fusion_layer = torch.nn.Conv1d(in_channels=config.hidden_size * 2, out_channels=config.hidden_size, kernel_size=3, padding=3//2)
            
        self.act_layer = ACT2FN[config.hidden_act]
        self.word_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.char_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, 
                char_hidden_states: torch.Tensor, 
                hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None, 
                head_mask: Optional[torch.Tensor] = None, 
                encoder_hidden_states: Optional[torch.Tensor] = None, 
                encoder_attention_mask: Optional[torch.Tensor] = None,
                output_hidden_states_word: Optional[bool] = None,
                output_hidden_states_char: Optional[bool] = None):
        all_hidden_states_word = () if output_hidden_states_word else None
        all_hidden_states_char = () if output_hidden_states_char else None
        all_attentions = ()
        # iterate over layers
        for i, layer_module in enumerate(self.layer):

            # add word and character hidden states to the output if needed
            if output_hidden_states_word:
                all_hidden_states_word = all_hidden_states_word + (hidden_states,)
            if output_hidden_states_char:
                all_hidden_states_char = all_hidden_states_char + (char_hidden_states,)

            # forward pass for word channel
            layer_outputs       = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            # forward pass for character channel
            char_layer_outputs  = layer_module(char_hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)

            word_outputs   = layer_outputs[0]
            char_outputs   = char_layer_outputs[0]

            # Heterogeneous Interaction (between word and character channels)
            # Feed forward -> CNN between word and character channels -> Feed forward + activation
            word_transform = self.word_linear1(word_outputs)
            char_transform = self.char_linear1(char_outputs)
            
            share_cat      = torch.cat([word_transform, char_transform], dim=-1)
            share_permute  = share_cat.permute(0, 2, 1)
            # CNN fusion layer
            share_fusion   = self.fusion_layer(share_permute)
            share_hidden   = share_fusion.permute(0, 2, 1)

            # divide step
            word_hidden_after_linear = self.word_linear2(share_hidden)
            char_hidden_after_linear = self.char_linear2(share_hidden)

            word_hidden = self.act_layer(word_hidden_after_linear)
            char_hidden = self.act_layer(char_hidden_after_linear)

            hidden_states  = self.word_norm(word_hidden + word_outputs)
            char_hidden_states = self.char_norm(char_hidden + char_outputs)

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # add word and character hidden states from the last layer to the output if needed
        if output_hidden_states_word:
            all_hidden_states_word = all_hidden_states_word + (hidden_states,)
        if output_hidden_states_char:
            all_hidden_states_char = all_hidden_states_char + (char_hidden_states,)

        outputs = (hidden_states, char_hidden_states)
        if output_hidden_states_word:
            outputs = outputs + (all_hidden_states_word,)
        if output_hidden_states_char:
            outputs = outputs + (all_hidden_states_char,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden word states), (all hidden char states), (all attentions)

class CharBertPreTrainedModel(BertPreTrainedModel):
    config_class = CharBertConfig
    base_model_prefix = "bert"

class CharBertModel(CharBertPreTrainedModel):
    def __init__(self, config: CharBertConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.char_embeddings = CharBertEmbeddings(config)
        self.encoder = CharBertEncoder(config)
        self.init_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, 
                char_input_ids: torch.Tensor, 
                start_ids: torch.Tensor, 
                end_ids: torch.Tensor, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None, 
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None, 
                head_mask: Optional[torch.Tensor] = None, 
                inputs_embeds: Optional[torch.Tensor] = None, 
                encoder_hidden_states: Optional[torch.Tensor] = None, 
                encoder_attention_mask: Optional[torch.Tensor] = None,
                output_hidden_states_word: Optional[bool] = None,
                output_hidden_states_char: Optional[bool] = None):
        """ Forward pass on the Model.

        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
        Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

        To behave as an decoder the model needs to be initialized with the
        `is_decoder` argument of the configuration set to `True`; an
        `encoder_hidden_states` is expected as an input to the forward pass.

        .. _`Attention is all you need`:
            https://arxiv.org/abs/1706.03762

        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(torch.long)  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape, attention_mask.shape))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError("Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(encoder_hidden_shape,
                                                                                                                               encoder_attention_mask.shape))

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, 
                                           position_ids=position_ids,
                                           token_type_ids=token_type_ids, 
                                           inputs_embeds=inputs_embeds)

        char_embeddings = self.char_embeddings(char_input_ids, start_ids, end_ids)

        encoder_outputs = self.encoder(char_embeddings,
                                       embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask,
                                       output_hidden_states_word=output_hidden_states_word,
                                       output_hidden_states_char=output_hidden_states_char)
        
        sequence_output, char_sequence_output = encoder_outputs[0], encoder_outputs[1]

        outputs = (sequence_output, char_sequence_output)
        if output_hidden_states_word or output_hidden_states_char or self.config.output_attentions:
            outputs = outputs + encoder_outputs[2:]
        return outputs  # sequence_output, char_sequence_output, (all hidden word states), (all hidden char states), (all attentions)


class BertLMPredictionHead(nn.Module):
    def __init__(self, config: CharBertConfig, term_vocab_size: int):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.adv_transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.adv_decoder = nn.Linear(config.hidden_size, term_vocab_size)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, 
                hidden_states: torch.Tensor, 
                char_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.transform(hidden_states)
        adv_hidden_states = self.adv_transform(char_hidden_states)
        mlm_scores = self.decoder(hidden_states) + self.bias
        term_scores = self.adv_decoder(adv_hidden_states)
        return mlm_scores, term_scores


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config: CharBertConfig, term_vocab_size: int):
        super().__init__()
        self.predictions = BertLMPredictionHead(config, term_vocab_size)

    def forward(self, 
                sequence_output: torch.Tensor, 
                char_sequence_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mlm_scores, term_scores = self.predictions(sequence_output, char_sequence_output)
        return mlm_scores, term_scores


class CharBertForMaskedLM(CharBertPreTrainedModel):
    def __init__(self, config: CharBertConfig, term_vocab_size: Optional[int] = None):
        super().__init__(config)

        self.term_vocab_size = term_vocab_size if term_vocab_size is not None else config.term_vocab_size

        self.bert = CharBertModel(config)
        self.cls = BertOnlyMLMHead(config, self.term_vocab_size)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(self, 
                char_input_ids: torch.Tensor, 
                start_ids: torch.Tensor, 
                end_ids: torch.Tensor, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None, 
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None, 
                head_mask: Optional[torch.Tensor] = None, 
                inputs_embeds: Optional[torch.Tensor] = None, 
                masked_lm_labels: Optional[torch.Tensor] = None, 
                adv_labels: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None, 
                encoder_attention_mask: Optional[torch.Tensor] = None,
                output_hidden_states_word: Optional[bool] = None,
                output_hidden_states_char: Optional[bool] = None):

        outputs = self.bert(char_input_ids,
                            start_ids,
                            end_ids,
                            input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            output_hidden_states_word=output_hidden_states_word,
                            output_hidden_states_char=output_hidden_states_char)

        token_sequence_repr = outputs[0]
        char_sequence_repr = outputs[1]

        prediction_scores, term_scores = self.cls(token_sequence_repr, char_sequence_repr)

        outputs = (prediction_scores, term_scores)
        if output_hidden_states_word or output_hidden_states_char or self.config.output_attentions:
            outputs = outputs + outputs[2:]  # Add hidden states and attention if they are here

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if adv_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            adv_term_loss = loss_fct(term_scores.view(-1, self.term_vocab_size), adv_labels.view(-1))
            outputs = (adv_term_loss,) + outputs

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, term_scores, (all hidden word states), (all hidden char states), (all attentions)


class CharBertForSequenceClassification(CharBertPreTrainedModel):
    def __init__(self, config: CharBertConfig):
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", 2)

        self.bert = CharBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.init_weights()

    def forward(self, 
                char_input_ids: torch.Tensor, 
                start_ids: torch.Tensor, 
                end_ids: torch.Tensor, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None, 
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None, 
                head_mask: Optional[torch.Tensor] = None, 
                inputs_embeds: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None,
                output_hidden_states_word: Optional[bool] = None,
                output_hidden_states_char: Optional[bool] = None):

        outputs = self.bert(char_input_ids,
                            start_ids,
                            end_ids,
                            input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_hidden_states_word=output_hidden_states_word,
                            output_hidden_states_char=output_hidden_states_char)

        token_sequence_repr = outputs[0]
        char_sequence_repr = outputs[1]
        sequence_repr = torch.cat([token_sequence_repr, char_sequence_repr], dim=-1)
        # average pooling from both token and character channels
        sequence_output = torch.mean(sequence_repr, dim=1)

        pooled_output = self.dropout(sequence_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,)
        if output_hidden_states_word or output_hidden_states_char or self.config.output_attentions:
            outputs = outputs + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                # regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # classification
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        # return outputs  # (loss), logits, (all hidden word states), (all hidden char states), (all attentions)
        return SequenceClassifierOutput(
            loss=loss if labels is not None else None,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

__all__ = [
    'CharBertEncoder',
    'CharBertModel',
    'CharBertForMaskedLM',
    'CharBertForSequenceClassification',
]