import copy
import random
import collections

import pandas as pd
import torch
import nltk
from transformers import PreTrainedTokenizerFast
import datasets

nltk.download('punkt_tab')
nltk.download('punkt')

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",["index", "label"])


class BERTDatasetPretraining(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        return item['url']


class CharBERTDatasetPretraining(torch.utils.data.Dataset):
    def __init__(self, 
                 tokenizer: PreTrainedTokenizerFast, 
                 char_vocab_path: str, 
                 term_vocab_path: str, 
                 dataset,
                 articles_to_read: int = 1000,
                 seq_len: int = 256,
                 random_seed: int = 42, 
                 adv_probability: float = 0.15,
                 mlm_probability: float = 0.1,
                 char_maxlen_for_word: int = 6,
                 world_size: int = 1,
                 world_rank: int = 0):
        """
        Args:
            tokenizer:              tokenizer
            char_vocab_path:        path to character vocabulary
            term_vocab_path:        path to term vocabulary
            file_path:              path to file
            input_nraws:            number of raw data blocks to read
            seq_len:                sequence length
            seed:                   random seed
            adv_probability:        probability of adversarial modification
            mlm_probability:        probability of masked language modeling
            char_maxlen_for_word:   maximum number of characters per token (char_maxlen_for_word = 6)
        """
        self.char2ids_dict = self.load_line_to_ids_dict(fname=char_vocab_path)
        self.term2ids_dict = self.load_line_to_ids_dict(fname=term_vocab_path)

        self.dataset = dataset
        self.total_articles = len(dataset)

        slice_articles = self.total_articles // world_size
        self.start_article = slice_articles * world_rank
        self.end_article = self.start_article + slice_articles
        if world_rank == world_size - 1:
            self.end_article = self.total_articles
        self.current_article_idx = self.start_article

        self.sequence_buffer = []
        self.articles_to_read = articles_to_read

        self.shuffle = True
        self.current_sample_idx = -1
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.rng = random.Random(random_seed)
        self.adv_probability = adv_probability
        self.mlm_probability = mlm_probability
        self.char_maxlen_for_word = char_maxlen_for_word

    def read_articles(self):
        start = self.current_article_idx
        end = start + self.articles_to_read
        if end > self.end_article:
            end = self.end_article
            # If we've reached the end, wrap around to the beginning
            if end - start < self.articles_to_read:
                remaining_articles = self.articles_to_read - (end - start)
                first_batch = self.dataset.select(range(start, end))
                second_batch = self.dataset.select(range(self.start_article, self.start_article + remaining_articles))
                articles = datasets.concatenate_datasets([first_batch, second_batch])
                self.current_article_idx = self.start_article + remaining_articles
            else:
                articles = self.dataset.select(range(start, end))
                self.current_article_idx = self.start_article
        else:
            articles = self.dataset.select(range(start, end))
            self.current_article_idx = end

        text = " ".join(articles['text'])

        doc_tokens = nltk.word_tokenize(text)

        # final tokens
        tokenized_tokens = []
        # mapping from sub-token index to original token
        sub_index_to_orig_token = {}
        # mapping from sub-token index to whether the sub-token is changed (True if changed, False otherwise)
        sub_index_to_change = {}
        # adversarial labels are for tokens that are part of the word or the original word that was modified
        # otherwise label is -1
        # character channel will predict the original word from the modified word or part of the word
        adv_labels = []
        for idx, token in enumerate(doc_tokens):
            ori_token = copy.deepcopy(token)
            # adversarially modify the word with a certain probability
            # drop, add or swap characters within the whole word
            if self.rng.random() < self.adv_probability:
                token = self.create_adv_word(token)
            sub_tokens = []
            # tokenize the word into sub-tokens
            sub_tokens = self.tokenizer.tokenize(token)
            for sub_w in sub_tokens:
                sub_index_to_orig_token[len(tokenized_tokens)] = token
                if ori_token != token:
                    sub_index_to_change[len(tokenized_tokens)] = True
                    # if the original word is in the term vocabulary, use the token id for the word
                    # otherwise, use the token id for the <unk> token
                    if ori_token.lower() in self.term2ids_dict:
                        adv_labels.append(self.term2ids_dict[ori_token.lower()])
                    else:
                        adv_labels.append(self.term2ids_dict['<unk>'])
                else:
                    sub_index_to_change[len(tokenized_tokens)] = False
                    adv_labels.append(-1)
                    
                tokenized_tokens.append(sub_w)

        # mask tokens that are not altered in any way
        input_tokens, mlm_labels = self.create_masked_lm_predictions(tokenized_tokens, sub_index_to_change)
        # convert tokens to ids
        tokenized_text = self.tokenizer.convert_tokens_to_ids(input_tokens)

        # seqlen - 2 for CLS and SEP
        seq_maxlen = self.seq_len - 2
        self.sequence_buffer = []

        # iterate over the text in blocks of seq_maxlen (drop the last block if it's less than seq_maxlen)
        for i in range(0, len(tokenized_text)-seq_maxlen+1, seq_maxlen): # Truncate in block of seq_len
            input_ids = self.tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i+seq_maxlen])
            mlm_input_labels = [-1] + mlm_labels[i:i+seq_maxlen] + [-1] #For CLS and SEP
            adv_input_labels = [-1] + adv_labels[i:i+seq_maxlen] + [-1]
            char_input_ids, start_ids, end_ids = self.build_char_inputs(input_ids, sub_index_to_orig_token, i, mlm_input_labels)
            assert len(input_ids) == len(mlm_input_labels)
            assert len(input_ids) == len(adv_input_labels)
            assert len(input_ids) == len(start_ids)
            assert len(input_ids) == len(end_ids)
            self.sequence_buffer.append((
                torch.tensor(char_input_ids), 
                torch.tensor(start_ids), 
                torch.tensor(end_ids),
                torch.tensor(input_ids), 
                torch.tensor(mlm_input_labels), 
                torch.tensor(adv_input_labels)
            ))
        self.current_sample_idx = -1
        if self.shuffle:
            random.shuffle(self.sequence_buffer)

    def create_masked_lm_predictions(self, tokens, sub_index_to_change):
        """
        Creates the predictions for the masked LM objective. 
        Mask only tokens that are not altered in any way.
        For tokens that are altered, the label is the token id for the modified token ([MASK], random or original token)
        For tokens that are not altered, the label is -1.
        """
        vocab_words = list(self.tokenizer.vocab.keys())
        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]" or token == "[PAD]":
                continue
            # Whole Word Masking means that if we mask all of the wordpieces
            # corresponding to an original word. When a word has been split into
            # WordPieces, the first token does not have any marker and any subsequence
            # tokens are prefixed with ##. So whenever we see the ## token, we
            # append it to the previous set of word indexes.
            #
            # Note that Whole Word Masking does *not* change the training code
            # at all -- we still predict each WordPiece independently, softmaxed
            # over the entire vocabulary.

            # add sub-tokens to the same word if they are prefixed with ##
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        # shuffle words (for masking) 
        self.rng.shuffle(cand_indexes)

        # output_tokens = list(tokens)
        output_tokens = tokens.copy()

        num_to_predict = int(round(len(tokens) * self.mlm_probability))

        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue

            # if the word (formed by tokens) has been altered in any way, skip it
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or sub_index_to_change[index]:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            # if the word (formed by tokens) has NOT been altered in any way mask the tokens in the word 
            # (80% of the time using [MASK], 10% of the time keep original, 10% of the time replace with random token)
            for index in index_set:
                covered_indexes.add(index)

                masked_token = None
                # 80% of the time, replace with [MASK]
                if self.rng.random() < 0.8:
                    masked_token = "[MASK]"
                else:
                    # 10% of the time, keep original
                    if self.rng.random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = vocab_words[self.rng.randint(0, len(vocab_words) - 1)]

                output_tokens[index] = masked_token
                masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

        assert len(masked_lms) <= num_to_predict
        # sort the sequence so it matches the order of the tokens
        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        masked_lm_labels = [-1] * len(tokens)
        for p in masked_lms:
            masked_lm_labels[p.index] = self.tokenizer.convert_tokens_to_ids(p.label)

        return output_tokens, masked_lm_labels

    def create_adv_word(self, orig_token):
        """
        Adversarially modify the word with a certain probability.
        Drop, add or swap characters within the whole word.
        """
        token = list(copy.deepcopy(orig_token))
        if len(orig_token) < 4:
            # TODO: maybe modify the 80, what characters to insert?
            rand_idx = self.rng.randint(0, 80)
            rand_char = list(self.char2ids_dict.keys())[rand_idx]
            insert_idx = self.rng.randint(0, len(orig_token)-1)
            token = token[:insert_idx] + [rand_char] + token[insert_idx:]
        else:
            if self.rng.random() < 0.5:
                rand_idx = self.rng.randint(0, len(orig_token)-1)
                del token[rand_idx]
            else:
                idx = random.randint(1, len(orig_token)-2)
                token[idx], token[idx+1] = token[idx+1], token[idx]
        token = ''.join(token)
        return token

    def build_char_inputs(self, input_ids, sub_index_to_ori_tok, start, labels):
        """
        input_ids:            input ids of tokens
        sub_index_to_ori_tok: sub-token to original token mapping
        start:                start index of the sequence
        labels:               MLM labels (token id for masked token, otherwise -1)
        """
        all_seq_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        char_ids = []
        start_ids = []
        end_ids = []
        # maximum length of the character input ids (char_maxlen_for_word = 6)
        char_maxlen = self.seq_len * self.char_maxlen_for_word
        for idx, token in enumerate(all_seq_tokens):
            if len(char_ids) >= char_maxlen:
                break
            token = token.strip("##")

            if token == self.tokenizer.unk_token:
                tok_orig_index = idx+start - 1
                if tok_orig_index in sub_index_to_ori_tok : #-1 for CLS
                    orig_token = sub_index_to_ori_tok[tok_orig_index]
                    #print(f'UNK: {token} to orig_tokens: {orig_token}')    
                    token = orig_token
            # character channel doesn't process special tokens (even masked token [MASK])
            if token in ["[CLS]", "[SEP]", "[MASK]", "[PAD]"] or labels[idx] != -1:
                start_ids.append(len(char_ids))
                end_ids.append(len(char_ids))
                char_ids.append(0)
            else:
                for char_idx, c in enumerate(token):
                    if len(char_ids) >= char_maxlen:
                        break
                    
                    if char_idx == 0:
                        start_ids.append(len(char_ids))
                    if char_idx == len(token) - 1:
                        end_ids.append(len(char_ids))

                    if c in self.char2ids_dict:
                        cid = self.char2ids_dict[c]
                    else:
                        cid = self.char2ids_dict["<unk>"]
                    char_ids.append(cid)

            if len(char_ids) < char_maxlen:
                char_ids.append(0)
        if len(char_ids) > char_maxlen:
            char_ids = char_ids[:char_maxlen]
        else:
            pad_len = char_maxlen - len(char_ids)
            char_ids = char_ids + [0] * pad_len
        while len(start_ids) < self.seq_len:
            start_ids.append(char_maxlen-1)
        while len(end_ids) < self.seq_len:
            end_ids.append(char_maxlen-1)

        return char_ids, start_ids, end_ids

    def load_line_to_ids_dict(self, fname):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        with open(fname, "r", encoding="utf-8") as reader:
            chars = reader.readlines()
        for index, char in enumerate(chars):
            char = char.rstrip('\n')
            vocab[char] = index
        return vocab

    def __len__(self):
        # harcoded length and we will rely on the early stopping
        return 100_000_000

    def __getitem__(self, item):
        self.current_sample_idx += 1

        if len(self.sequence_buffer) == 0 or self.current_sample_idx == len(self.sequence_buffer):
            self.read_articles()
            self.current_sample_idx += 1
        # examples are in the form:
        # (char_input_ids, start_ids, end_ids, input_ids, labels, adv_input_labels)
        # char_input_ids:       character input ids 
        #                       - ids from the character vocabulary (0 for padding)
        # start_ids:            start ids 
        #                       - indices of the first character of the token in the character input ids
        # end_ids:              end ids 
        #                       - indices of the last character of the token in the character input ids
        # input_ids:            input ids 
        #                       - ids from the token vocabulary
        # labels:               MLM labels 
        #                       - token id for masked token, otherwise -1
        # adv_input_labels:     adversarial input labels 
        #                       - character id for modified token, otherwise -1
        return self.sequence_buffer[self.current_sample_idx]