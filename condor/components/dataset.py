import random
import torch
import numpy as np
import copy

from torch.utils.data.dataset import Dataset

from condor.components.tokenizer import CharacterTokenizer
from condor.util import generate_x_y, decode


class SpacingDataset(Dataset):
    def __init__(self, tok: CharacterTokenizer, sents, max_len):
        self.tok = tok
        self.max_len = max_len
        self.data = sents
        self.pad_id = tok.token_to_idx(tok._pad_token)
        self.ignore_index = 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # tgt
        tgt = self.data[idx]

        # sample src
        text_wo_space = tgt.replace(' ', '')
        src_id = self._tokenize(self.tok, self.max_len, text_wo_space)
        _, src_space = generate_x_y(text_wo_space)
        _, target = generate_x_y(tgt)

        change_cnt = random.sample(range(sum(src_space) + 1), 1)[0]
        change_idx = random.sample([i for i, v in enumerate(src_space) if v == 1], change_cnt)
        if change_idx:
            for i in change_idx:
                src_space[i] = 0

        if len(target) < self.max_len:
            target = target + [self.ignore_index] * (self.max_len - len(target))
            src_space = src_space + [self.ignore_index] * (self.max_len - len(target))
        else:
            target = target[:self.max_len]
            src_space = src_space[:self.max_len]
        return np.array(src_id), np.array(src_space), np.array(target)

    def _tokenize(self, tokenizer, max_len, sent):
        tokens = tokenizer.tokenize(sent)

        if len(tokens) < max_len:
            tokens = tokens + [self.pad_id] * (max_len - len(tokens))
        elif len(tokens) == 0:
            tokens += [0]
        else:
            tokens = tokens[:max_len]
        return tokens
