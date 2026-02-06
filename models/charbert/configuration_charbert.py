from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.models.bert.configuration_bert import BertConfig


class CharBertConfig(PretrainedConfig):

    model_type = "charbert"

    def __init__(self,
                 bert_config: Optional[BertConfig] = None,
                 vocab_size: Optional[int] = None,
                 max_position_embeddings: Optional[int] = 256,
                 pad_token_id: Optional[int] = 0,
                 char_vocab_size: Optional[int] = 1001,
                 char_embedding_size: Optional[int] = 128,
                 term_vocab_size: Optional[int] = 30_000,
                 **kwargs):
        if bert_config is None:
            bert_config = BertConfig()

        super().__init__(pad_token_id=pad_token_id, **kwargs)
         
        self.vocab_size = vocab_size if vocab_size is not None else getattr(self, "vocab_size", getattr(bert_config, "vocab_size", None))
        self.max_position_embeddings = max_position_embeddings

        self.hidden_size = getattr(self, "hidden_size", bert_config.hidden_size)
        self.num_hidden_layers = getattr(self, "num_hidden_layers", bert_config.num_hidden_layers)
        self.num_attention_heads = getattr(self, "num_attention_heads", bert_config.num_attention_heads)
        self.hidden_act = getattr(self, "hidden_act", bert_config.hidden_act)
        self.intermediate_size = getattr(self, "intermediate_size", bert_config.intermediate_size)
        self.hidden_dropout_prob = getattr(self, "hidden_dropout_prob", bert_config.hidden_dropout_prob)
        self.attention_probs_dropout_prob = getattr(self, "attention_probs_dropout_prob", bert_config.attention_probs_dropout_prob)
        self.type_vocab_size = getattr(self, "type_vocab_size", bert_config.type_vocab_size)
        self.initializer_range = getattr(self, "initializer_range", bert_config.initializer_range)
        self.layer_norm_eps = getattr(self, "layer_norm_eps", bert_config.layer_norm_eps)
        self.position_embedding_type = getattr(self, "position_embedding_type", bert_config.position_embedding_type)
        self.use_cache = getattr(self, "use_cache", bert_config.use_cache)
        self.classifier_dropout = getattr(self, "classifier_dropout", bert_config.classifier_dropout)

        # specific config for CharBERT
        self.char_vocab_size = char_vocab_size
        self.char_embedding_size = char_embedding_size
        self.term_vocab_size = term_vocab_size

        # specifically telling to tie word embeddings and decoder embeddings for MLM task
        self.tie_word_embeddings = True

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
                f'}}')

__all__ = ["CharBertConfig"]