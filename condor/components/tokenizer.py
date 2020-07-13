#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division


import dill

from tqdm import tqdm
from collections import Counter

# ==================================================


class CharacterTokenizer:
    def __init__(
        self,
        model_path: str = None,
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        unknown_token: str = "<unk>",
        start_token: str = "<s>",
        end_token: str = "</s>",
    ):

        self._unknown_token = unknown_token
        self._pad_token = pad_token
        self._cls_token = cls_token
        self._start_token = start_token
        self._sep_token = sep_token
        self._end_token = end_token

        if model_path:
            self.load(model_path)
            self.tok_to_id_dict = self.model["dict"]
            self.id_to_tok_dict = {}
            for k, v in self.tok_to_id_dict.items():
                self.id_to_tok_dict[v] = k
            self.c_list = list(self.tok_to_id_dict.keys())

        else:
            self.model = {"dict": {}, "model_name": ""}
            self.tok_to_id_dict = {}
            self.id_to_tok_dict = {}
            self.c_list = []

        self.model_name = ""

    def load(self, model_path):
        with open(model_path, "rb") as file:
            model = dill.load(file)

        if set(model.keys()) != {"dict", "model_name"}:
            raise KeyError("Model keys should be 'dict' and 'model_name'")
        else:
            self.model = model
            self.tok_to_id_dict = model["dict"]
            self.id_to_tok_dict = {}
            for k, v in self.tok_to_id_dict.items():
                self.id_to_tok_dict[v] = k
            self.c_list = list(self.tok_to_id_dict.keys())

    def tokenize(self, text, to_id=True):
        if to_id:
            res = list(text)
            res = [
                self.tok_to_id_dict[t]
                if t in self.c_list
                else self.tok_to_id_dict[self._unknown_token]
                for t in res
            ]
            return res
        else:
            return list(text)

    def token_to_text(self, token):
        return "".join(token)

    def idx_to_token(self, idx):
        res = [self.id_to_tok_dict[i] if i != self._unknown_token else "" for i in idx]
        return res

    def token_to_idx(self, token):
        if token in self.c_list:
            return self.tok_to_id_dict[token]
        else:
            return self.tok_to_id_dict[self._unknown_token]

    def idx_to_text(self, idx):
        res = [self.id_to_tok_dict[i] if i != self._unknown_token else "" for i in idx]
        res = "".join(res)
        return res

    def text_to_idx(self, text: str, max_seq_len: int = None, use_pad: bool = False):

        idx = self.tokenize(text)
        if max_seq_len is None:
            max_seq_len = len(idx)
        if use_pad:
            idx += [self.token_to_idx(self._pad_token)] * (max_seq_len - len(idx))
        return idx[:max_seq_len]

    def train(
        self,
        model_prefix: str,
        sents: list = None,
        input_path: str = None,
        min_count: int = 3,
    ):
        if not sents and not input_path:
            raise ValueError("One of input path or sentence list should be given")

        if input_path:
            with open(input_path, "r", encoding="utf-8") as file:
                corpus = file.read()
            corpus = corpus.split("\n")

        if sents:
            corpus = sents

        char = []
        for s in tqdm(corpus, desc="Adding characters"):
            char += list(s)
        counter = Counter(char)

        pre_token = [
            self._unknown_token,
            self._start_token,
            self._end_token,
            self._pad_token,
            self._cls_token,
            self._sep_token,
        ]

        char2idx = {}
        for v, k in enumerate(pre_token):
            char2idx[k] = v

        for i, key in enumerate(counter):
            if counter[key] >= min_count:
                char2idx[key] = len(char2idx)

        model_name = model_prefix + ".model"

        outp = {"dict": char2idx, "model_name": model_prefix}
        with open(model_name, "wb") as file:
            dill.dump(outp, file)

    def __repr__(self):
        unk = '"{}"'.format(self._unknown_token) if self._unknown_token else "None"
        return "Vocab(size={}, unk={}, pad={})".format(
            len(self.tok_to_id_dict), unk, self._pad_token
        )

    def __len__(self):
        return len(self.tok_to_id_dict)
