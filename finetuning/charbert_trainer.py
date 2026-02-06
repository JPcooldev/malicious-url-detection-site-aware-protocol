import collections
import argparse
from typing import Optional, Tuple, List, Dict

from transformers import PreTrainedTokenizerFast
import torch

from .trainer import TrainerBase
from finetuning.data_processing.dataset import URLDataset


class CharBertTrainer(TrainerBase):
    """
    Trainer for CharBERT-based models
    """

    tokenizer: PreTrainedTokenizerFast

    def __init__(self, 
        args: argparse.Namespace, 
        model: torch.nn.Module, 
        tokenizer: PreTrainedTokenizerFast, 
        dataset: URLDataset, 
        device: Optional[torch.device] = None, 
        move_to_device: bool = True, 
        log_to_mlflow: bool = True,
        is_evaluation: bool = False):
        super().__init__(
            args, model, dataset, log_to_mlflow=log_to_mlflow, device=device, move_to_device=move_to_device, is_evaluation=is_evaluation
        )
        self.tokenizer = tokenizer
        self.char2ids_dict = self.load_char_to_ids_dict()

    def prepare_batch(self, batch: Tuple[List[str], List[int]]):
        urls, labels = zip(*batch)
        encoded_batch = self.tokenizer(urls, padding=True, truncation=True, max_length=self.args.max_sequence_length)
        char_input_ids, start_ids, end_ids = self.build_char_inputs(encoded_batch["input_ids"])
        input_dict = {
            "input_ids": torch.tensor(encoded_batch["input_ids"]),
            "attention_mask": torch.tensor(encoded_batch["attention_mask"]),
            "token_type_ids": torch.tensor(encoded_batch["token_type_ids"]),
            "char_input_ids": char_input_ids,
            "start_ids": start_ids,
            "end_ids": end_ids,
        }
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return input_dict, labels_tensor

    def build_char_inputs(self, input_ids: List[List[int]]):
        """Build character inputs for CharBERT-based models"""
        batch_char_ids = []
        batch_start_ids = []
        batch_end_ids = []
        block_size = len(input_ids[0])
        char_maxlen = block_size * self.args.char_max_length_per_token

        for ids in input_ids:
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            char_ids = []
            start_ids = []
            end_ids = []
            for token in tokens:
                if len(char_ids) >= char_maxlen:
                    break
                token = token.strip('##')
                if token == self.tokenizer.unk_token or token in ["[CLS]", "[SEP]", "[PAD]"]:
                    # add space as a character
                    start_ids.append(len(char_ids))
                    end_ids.append(len(char_ids))
                    char_ids.append(0)
                else:
                    for char_idx, char in enumerate(token):
                        if len(char_ids) >= char_maxlen:
                            break
                        if char_idx == 0:
                            start_ids.append(len(char_ids))
                        if char_idx == len(token) - 1:
                            end_ids.append(len(char_ids))
                        
                        if char in self.char2ids_dict:
                            char_id = self.char2ids_dict[char]
                        else:
                            char_id = self.char2ids_dict['<unk>']
                        char_ids.append(char_id)
                # add space between tokens
                if len(char_ids) < char_maxlen:
                    char_ids.append(0)

            if len(char_ids) > char_maxlen:
                char_ids = char_ids[:char_maxlen]
            else:
                pad_len = char_maxlen - len(char_ids)
                char_ids = char_ids + [0] * pad_len
            while len(start_ids) < block_size:
                start_ids.append(char_maxlen-1)
            while len(end_ids) < block_size:
                end_ids.append(char_maxlen-1)

            batch_char_ids.append(char_ids)
            batch_start_ids.append(start_ids)
            batch_end_ids.append(end_ids)
        
        batch_char_ids = torch.tensor(batch_char_ids)
        batch_start_ids = torch.tensor(batch_start_ids)
        batch_end_ids = torch.tensor(batch_end_ids)
        return batch_char_ids, batch_start_ids, batch_end_ids

    def load_char_to_ids_dict(self) -> Dict[str, int]:
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        with open(self.args.char_vocab_path, "r", encoding="utf-8") as reader:
            chars = reader.readlines()
        for index, char in enumerate(chars):
            char = char.rstrip('\n')
            vocab[char] = index
        return vocab
