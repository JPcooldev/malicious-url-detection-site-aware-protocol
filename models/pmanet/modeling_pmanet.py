# This file is modified from https://github.com/Alixyvtte/Malicious-URL-Detection-PMANet
# Original code copyright (c) Alixyvtte, licensed under Apache License 2.0.

from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from models.charbert.modeling_charbert import CharBertModel, CharBertPreTrainedModel
from models.pmanet.configuration_pmanet import PMANetConfig

from transformers.modeling_outputs import SequenceClassifierOutput


class LayerAwareAttentionModule(nn.Module):
    """
        Initializes LayerAwareAttentionModule

        :param channel: The number of input channels.
        :param reduction: The reduction ratio for channel attention (default is 3).
        :param spatial_kernel: The size of the spatial kernel (default is 7).
    """
    def __init__(self, channel: int, reduction: int = 3):
        super().__init__()

        # Channel attention: Compress H and W to 1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Shared MLP
        mid_channel = max(1, channel // reduction)
        self.mlp = nn.Sequential(
            # Use Conv2d for more convenient operations compared to Linear
            nn.Conv2d(channel, mid_channel, 1, bias=False),
            nn.ReLU(inplace=True),  # Inplace operation to save memory
            nn.Conv2d(mid_channel, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        return x


class PMANetForSequenceClassification(CharBertPreTrainedModel):
    def __init__(self, config: PMANetConfig):
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", 2)

        self.bert = CharBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.fuse = nn.Conv1d(2 * config.hidden_size, config.hidden_size, kernel_size=1)
        self.layer_aware_attention = LayerAwareAttentionModule(channel=self.config.num_hidden_layers)

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
                output_hidden_states_word: Optional[bool] = True,
                output_hidden_states_char: Optional[bool] = True):
        
        device = char_input_ids.device

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

        all_hidden_states_word = outputs[2]
        all_hidden_states_char = outputs[3]
        bert_outputs = outputs

        fuse_output = []
        # iterate over all hidden states except the input embeddings
        for word_hidden_state, char_hidden_state in zip(all_hidden_states_word[:-1], all_hidden_states_char[:-1]):
            word_hidden_state = word_hidden_state.to(device)
            char_hidden_state = char_hidden_state.to(device)

            # Concatenate two tensors
            hidden_state = torch.cat([word_hidden_state, char_hidden_state], dim=-1)
            # Fuse with the convolutional layer
            fused_hidden_state = self.fuse(hidden_state.transpose(1, 2))  # Transpose dimensions to match the convolutional layer's input requirements
            fused_hidden_state = fused_hidden_state.transpose(1, 2)  # fused_hidden_state size: batch_size, seq_len, hidden_size
            fuse_output.append(fused_hidden_state)
            
        pyramid = torch.stack(fuse_output, dim=0).permute(1, 0, 2, 3)
        # pyramid size: batch_size, num_layers, seq_len, hidden_size

        pos_pooled = self.layer_aware_attention(pyramid)
        # pos_pooled size: batch_size, num_layers, seq_len, hidden_size

        # Initialize a list to store pyramid pooling results
        pooled_features = []

        for level in self.config.pyramid_levels:
            # Calculate the pooling window size for each level
            window_size = max(1, pos_pooled.size(1) // level)

            # Use average pooling for each level
            pooled_feature_tensor = F.avg_pool2d(
                pos_pooled.permute(0, 3, 2, 1), (1, window_size)
            ).permute(0, 3, 2, 1)
            # pooled_tensor_size: batch_size, num_layers, seq_len, hidden_size

            # Add the pooling results for each level to the list
            pooled_features.append(pooled_feature_tensor)

        # Concatenate the pyramid pooling results along the feature dimension (pooled levels)
        concatenated_features = torch.cat(pooled_features, dim=1)

        # Compress the features (first accros seq_len and then accros pooled levels)
        compressed_feature_tensor = torch.mean(concatenated_features, dim=2)
        compressed_feature_tensor = torch.mean(compressed_feature_tensor, dim=1)

        compressed_features = self.dropout(compressed_feature_tensor)
        logits = self.classifier(compressed_features)

        outputs = (logits,)
        if output_hidden_states_word or output_hidden_states_char or self.config.output_attentions:
            outputs = outputs + bert_outputs[2:]  # add hidden states and attention if they are here

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

__all__ = ["PMANetForSequenceClassification"]