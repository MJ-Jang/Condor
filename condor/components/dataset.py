import random
import torch
import numpy as np

from torch.utils.data.dataset import Dataset

from condor.components.tokenizer import CharacterTokenizer
from condor.util import generate_x_y, decode


class SpacingDataset(Dataset):
    def __init__(self, tok: CharacterTokenizer, sents, max_len, noise_prob=0.2):
        self.tok = tok
        self.max_len = max_len
        self.data = sents
        self.pad_id = tok.token_to_idx(tok._pad_token)
        self.noise_prob = noise_prob
        self.ignore_index = 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        char_list, target = generate_x_y(sent)
        inputs = [self.tok.token_to_idx(t) for t in char_list]
        if random.random() <= self.noise_prob:
            current_space = self.generate_noise(target, self.noise_prob)
        else:
            current_space = target

        if len(inputs) <= self.max_len:
            inputs += [self.pad_id] * (self.max_len - len(inputs))
            target += [2] * (self.max_len - len(target))
            current_space += [2] * (self.max_len - len(current_space))
        else:
            inputs = inputs[:self.max_len]
            target = target[:self.max_len]
            current_space = current_space[:self.max_len]

        return np.array(inputs), np.array(current_space), np.array(target)

    @staticmethod
    def generate_noise(space_idxs: list, noise_prob: float = 0.2):
        outp = []
        for i in space_idxs:
            if random.random() <= noise_prob:
                if i == 0:
                    outp.append(1)
                elif i == 1:
                    outp.append(0)
            else:
                outp.append(i)
        return outp
