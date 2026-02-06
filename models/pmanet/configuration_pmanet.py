from typing import Optional

from transformers.models.bert.configuration_bert import BertConfig

from models.charbert.configuration_charbert import CharBertConfig


class PMANetConfig(CharBertConfig):
    
    model_type = "pmanet"

    def __init__(self,
                 bert_config: Optional[BertConfig] = None,
                 vocab_size: Optional[int] = None,
                 max_position_embeddings: Optional[int] = 256,
                 pad_token_id: Optional[int] = 0,
                 char_vocab_size: Optional[int] = 1001,
                 char_embedding_size: Optional[int] = 128,
                 term_vocab_size: Optional[int] = 30_000,
                 pyramid_levels: Optional[list[int]] = [1, 2],
                 **kwargs):
        super().__init__(
            bert_config=bert_config, 
            vocab_size=vocab_size, 
            max_position_embeddings=max_position_embeddings, 
            pad_token_id=pad_token_id, 
            char_vocab_size=char_vocab_size, 
            char_embedding_size=char_embedding_size, 
            term_vocab_size=term_vocab_size,
            **kwargs)

        if pyramid_levels is None:
            n_layers = self.num_hidden_layers
            default_levels = [1, 2] if n_layers >= 2 else [1]
            pyramid_levels = [l for l in default_levels if l <= n_layers]

        for level in pyramid_levels:
            assert level > 0, "Pyramid levels must be greater than 0"
            assert level <= self.num_hidden_layers, "Pyramid levels must be <= to the number of hidden layers"

        # specific config for PMANet
        self.pyramid_levels = pyramid_levels

    
    def __repr__(self):
        """Override __repr__ to avoid the default transformers serialization that causes issues."""
        return (f'{self.__class__.__name__} {{\n'
                f'  "vocab_size": {self.vocab_size},\n'
                f'  "hidden_size": {self.hidden_size},\n'
                f'  "num_hidden_layers": {self.num_hidden_layers},\n'
                f'  "num_attention_heads": {self.num_attention_heads},\n'
                f'  "intermediate_size": {self.intermediate_size},\n'
                f'  "hidden_act": "{self.hidden_act}",\n'
                f'  "hidden_dropout_prob": {self.hidden_dropout_prob},\n'
                f'  "attention_probs_dropout_prob": {self.attention_probs_dropout_prob},\n'
                f'  "max_position_embeddings": {self.max_position_embeddings},\n'
                f'  "type_vocab_size": {self.type_vocab_size},\n'
                f'  "initializer_range": {self.initializer_range},\n'
                f'  "layer_norm_eps": {self.layer_norm_eps},\n'
                f'  "position_embedding_type": "{self.position_embedding_type}",\n'
                f'  "use_cache": {"true" if self.use_cache else "false"},\n'
                f'  "classifier_dropout": {"null" if self.classifier_dropout is None else self.classifier_dropout},\n'
                f'  "pad_token_id": {self.pad_token_id},\n'
                f'  "char_vocab_size": {self.char_vocab_size},\n'
                f'  "char_embedding_size": {self.char_embedding_size},\n'
                f'  "term_vocab_size": {self.term_vocab_size},\n'
                f'  "pyramid_levels": {self.pyramid_levels},\n'
                f'}}')

__all__ = ["PMANetConfig"]