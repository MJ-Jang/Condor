import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Classifier(nn.Module):
    def __init__(self, d_model, class_num, d_ff=128, dropout=0.5):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, class_num)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class TransformerSpaceCorrector(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 n_head: int,
                 n_layers: int,
                 dim_ff: int,
                 dropout: float,
                 pad_id: int):
        super(TransformerSpaceCorrector, self).__init__()

        self.vocab_size = vocab_size
        self.label_size = 2+1

        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.label_embedding = nn.Embedding(self.label_size, d_model)

        self.position_embedding = PositionalEncoding(d_model, dropout)
        enc_layer = TransformerEncoderLayer(d_model, n_head, dim_ff)
        # enc_norm = LayerNorm(d_model)
        # self.encoder = TransformerEncoder(enc_layer, n_layers, enc_norm)
        self.encoder = TransformerEncoder(enc_layer, n_layers)

        self.classifier = Classifier(d_model=d_model, class_num=2, d_ff=128, dropout=dropout)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src_id, space_id):
        '''
        src_id: input token index sequence: N * S
        space_id: current spacing status: N * S
        '''
        if len(src_id.size()) == 1:
            src_id = src_id.reshape(1, -1)

        x = self.embedding(src_id) * math.sqrt(self.d_model)  # (N * S * E)
        x = self.position_embedding(x)
        x = x.transpose(1, 0)  # (S * N * E)

        s = self.label_embedding(space_id) * math.sqrt(self.d_model)
        s = self.position_embedding(s)
        s = s.transpose(1, 0)
        x = x + s

        seq_mask = self._generate_square_subsequent_mask(len(x)).to(x.device)

        # S * S
        x = self.encoder(x, seq_mask).transpose(1, 0)  # (N * S * E)
        logits = self.classifier(x)
        return logits
