import os
from typing import Literal, Optional, Tuple
import json

import torch
import torch.nn.init as init
import mlflow
from transformers import (
    AutoModelForMaskedLM, 
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    PreTrainedTokenizerFast,
)

from .charbert.configuration_charbert import CharBertConfig
from .charbert.modeling_charbert import CharBertForMaskedLM, CharBertForSequenceClassification
from .pmanet.configuration_pmanet import PMANetConfig
from .pmanet.modeling_pmanet import PMANetForSequenceClassification


SIZE_TO_CHECKPOINT = {
    "tiny": "google/bert_uncased_L-2_H-128_A-2",
    "mini": "google/bert_uncased_L-4_H-256_A-4",
    "small": "google/bert_uncased_L-4_H-512_A-8"
}

SIZE_TO_CONFIG = {
    "tiny": AutoConfig.from_pretrained(SIZE_TO_CHECKPOINT["tiny"]),
    "mini": AutoConfig.from_pretrained(SIZE_TO_CHECKPOINT["mini"]),
    "small": AutoConfig.from_pretrained(SIZE_TO_CHECKPOINT["small"])
}

def init_model_pretraining(
        model_type: Literal["bert-url", "urlbert", "charbert"], 
        model_size: Literal["tiny", "mini", "small"], 
        vocab_size: int,
        max_position_embeddings: int = 256, 
        char_vocab_size: Optional[int] = 1_000,
        term_vocab_size: Optional[int] = 30_000,
        char_embedding_size: Optional[int] = 128):
    assert model_type in ["bert-url", "urlbert", "charbert"], f"Invalid model type: {model_type}, choose from ['bert-url', 'urlbert', 'charbert']"
    assert model_size in ["tiny", "mini", "small"], f"Invalid model size: {model_size}, choose from ['tiny', 'mini', 'small']"

    if model_type == "bert-url" or model_type == 'urlbert':
        config = SIZE_TO_CONFIG[model_size]
        config.vocab_size = vocab_size
        config.max_position_embeddings = max_position_embeddings
        model = AutoModelForMaskedLM.from_config(SIZE_TO_CONFIG[model_size])

    elif model_type == "charbert":
        config = CharBertConfig(
            bert_config=SIZE_TO_CONFIG[model_size],
            vocab_size=vocab_size,
            pad_token_id=0,
            max_position_embeddings=max_position_embeddings,
            char_vocab_size=char_vocab_size,
            char_embedding_size=char_embedding_size,
            term_vocab_size=term_vocab_size,
        )
        model = CharBertForMaskedLM.from_pretrained(SIZE_TO_CHECKPOINT[model_size], config=config, ignore_mismatched_sizes=True)
        bert_model = AutoModelForMaskedLM.from_pretrained(SIZE_TO_CHECKPOINT[model_size])
        # copy the first max_position_embeddings position embeddings from BERT to CharBERT
        with torch.no_grad():
            bert_pos = bert_model.bert.embeddings.position_embeddings.weight.data
            model.bert.embeddings.position_embeddings.weight.copy_(bert_pos[:max_position_embeddings])

            # classical init sometimes resulted with NaNs in params
            for name, param in model.bert.char_embeddings.rnn_layer.named_parameters():
                if torch.isnan(param).any():
                    param.data = torch.nan_to_num(param.data, nan=0.0) 
                    print(f"NaNs found in {name}, setting to 0.0")
            # classical init resulted in huge numbers
            model.bert.encoder.fusion_layer.apply(init_weights)
            model.bert.char_embeddings.rnn_layer.apply(init_weights)

    return model


def init_weights(m):
    if isinstance(m, torch.nn.Conv1d):
        init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    if isinstance(m, torch.nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                # input-hidden weights (for reset, update, new gates)
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # hidden-hidden weights
                init.orthogonal_(param.data)
            elif 'bias' in name:
                # initialize all biases to zero
                param.data.fill_(0)
                # set forget gate bias to 1 to help learning stability
                # GRU gates are reset, update, new; update gate corresponds to second third of biases
                n = param.size(0)
                # biases are concatenated for 3 gates, so set middle bias block to 1
                start = n // 3
                end = 2 * n // 3
                param.data[start:end].fill_(1)


def load_tokenizer(model_type: Literal["bert-url", "bert-text", "urlbert", "charbert", "pmanet"]):
    
    assert model_type in ["bert-url", "bert-text", "urlbert", "charbert", "pmanet"], f"Invalid model type: {model_type}, choose from ['bert-url', 'bert-text', 'urlbert', 'charbert', 'pmanet']"
    
    if model_type == "bert-text" or model_type == "charbert" or model_type == "pmanet":
        # tiny, mini and small BERTs all use BERT tokenizer
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    elif model_type == "bert-url" or model_type == "urlbert":
        tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tokenizers", "wordpiece-url-vocab-15360")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer


def load_pretrained_model(
    model_type: Literal["bert-url", "bert-text", "urlbert", "charbert", "pmanet"], 
    model_size: Literal["tiny", "mini"],
    run_id: str = None):

    assert model_type in ["bert-url", "bert-text", "urlbert", "charbert", "pmanet"], f"Unsupported architecure. Choose from ['bert-url', 'bert-text', 'urlbert', 'charbert', 'pmanet']"
    assert model_size in ["tiny", "mini", "small", "base"], "Unsupported model size. Choose from ['tiny', 'mini', 'small', 'base']"
    assert run_id is not None and model_type in ["charbert", "pmanet"], "Run ID is required for charbert and pmanet"
    
    if model_type == "bert-text":
        checkpoint = SIZE_TO_CHECKPOINT[model_size]
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    else:
        model = mlflow_download_model(model_type, model_size, run_id)
    return model


def mlflow_download_model(
    model_type: Literal["bert-text", "bert-url", "urlbert", "charbert", "pmanet"], 
    model_size: Literal["tiny", "mini"], 
    run_id: str):

    experiment_path = os.getenv("MLFLOW_PRETRAINING_EXPERIMENT_NAME")
    mlflow.set_experiment(experiment_path)

    artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id)
    model_local_path = os.path.join(artifact_path, 'pretrained_model', 'model')

    if model_type == "bert-url" or model_type == "urlbert":
        model = AutoModelForSequenceClassification.from_pretrained(model_local_path)
    elif model_type == "charbert":
        model = CharBertForSequenceClassification.from_pretrained(model_local_path)
    elif model_type == "pmanet":
        bert_config = AutoConfig.from_pretrained(SIZE_TO_CHECKPOINT[model_size])
        charbert_config = CharBertConfig.from_pretrained(model_local_path)
        pmanet_config = PMANetConfig(
            bert_config=bert_config,
            vocab_size=charbert_config.vocab_size,
            max_position_embeddings=charbert_config.max_position_embeddings,
            pad_token_id=charbert_config.pad_token_id,
            char_vocab_size=charbert_config.char_vocab_size,
            char_embedding_size=charbert_config.char_embedding_size,
            term_vocab_size=charbert_config.term_vocab_size,
            pyramid_levels=[1, 2],
        )
        model = PMANetForSequenceClassification.from_pretrained(model_local_path, config=pmanet_config)
        _pmanet_post_load_init(model)

    experiment_path = os.getenv("MLFLOW_CROSSVALIDATION_EXPERIMENT_NAME")
    mlflow.set_experiment(experiment_path)

    return model


def _pmanet_post_load_init(model):
    with torch.no_grad():
        # Fuse conv (Conv1d): Kaiming uniform, zero bias
        if hasattr(model, "fuse") and isinstance(model.fuse, torch.nn.Conv1d):
            init.kaiming_uniform_(model.fuse.weight, a=0.0, mode="fan_in", nonlinearity="relu")
            if model.fuse.bias is not None:
                init.zeros_(model.fuse.bias)

        # Layer-aware attention shared MLP convs (Conv2d): Kaiming uniform
        attn = getattr(model, "layer_aware_attention", None)
        mlp = getattr(attn, "mlp", None)
        if mlp is not None:
            for layer in mlp:
                if isinstance(layer, torch.nn.Conv2d):
                    init.kaiming_uniform_(layer.weight, a=0.0, mode="fan_in", nonlinearity="relu")
                    if layer.bias is not None:
                        init.zeros_(layer.bias)


def get_finetuned_model_run_id(model_type: str, model_size: str) -> str:
    experiment = mlflow.get_experiment_by_name(os.getenv("MLFLOW_CROSSVALIDATION_EXPERIMENT_NAME"))
    parent_runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    parent_run_id = parent_runs[(parent_runs['tags.mlflow.runName'].str.contains(f'{model_type}-{model_size}', na=False, case=False))]['run_id'].values[0]
    run_id = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}' and tags.mlflow.runName = 'fold_0'",
    ).run_id.values[0]
    return run_id
    

def mlflow_load_finetuned_model_and_tokenizer(
    model_type: str, 
    model_size: str, 
    artifact_path="classification_model", 
    model_artifact_path="model") -> Tuple[torch.nn.Module, PreTrainedTokenizerFast]:

    run_id = get_finetuned_model_run_id(model_type, model_size)
    artifact_download = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)

    model_path = os.path.join(artifact_download, model_artifact_path)
    with open(os.path.join(artifact_download, model_artifact_path, 'config.json'), 'r') as f:
        model_config = json.load(f) 
    architecture = model_config['architectures'][0]

    if 'PMANetForSequenceClassification' in architecture:
        model = PMANetForSequenceClassification.from_pretrained(model_path)
    elif 'CharBertForSequenceClassification' in architecture:
        model = CharBertForSequenceClassification.from_pretrained(model_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model