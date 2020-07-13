import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dill

from condor.components.module import TransformerSpaceCorrector
from condor.components.tokenizer import CharacterTokenizer
from condor.components.dataset import SpacingDataset
from condor.util import generate_x_y, decode
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict


class KorSpaceCorrector:
    def __init__(self,
                 tok: CharacterTokenizer,
                 threshold: float = 0.6,
                 d_model: int = 256,
                 n_head: int = 4,
                 n_layers: int = 2,
                 dim_ff: int = 256,
                 dropout: float = 0.5,
                 use_gpu: bool = True):
        self.device = 'cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu'
        if self.device == 'cuda:0':
            self.n_gpu = torch.cuda.device_count()
        else:
            self.n_gpu = 0

        self.tok = tok
        self.pad_id = self.tok.token_to_idx(self.tok._pad_token)
        self.model_conf = {'vocab_size': len(self.tok),
                           'd_model': d_model,
                           'n_head': n_head,
                           'n_layers': n_layers,
                           'dim_ff': dim_ff,
                           'dropout': dropout,
                           'pad_id': self.pad_id}

        self.model = TransformerSpaceCorrector(**self.model_conf)
        self.threshold = threshold
        self.softmax = torch.nn.Softmax(dim=-1)

        if self.n_gpu == 1:
            self.model.cuda()
        elif self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.cuda()

    def correct(self, text: str):
        x, space_id = generate_x_y(text)
        src_id = self.tok.tokenize(''.join(x))
        src_id = torch.LongTensor([src_id]).to(self.device)
        space_id_tensor = torch.LongTensor([space_id]).to(self.device)

        logits = self.model(src_id, space_id_tensor).detach()
        logits = self.softmax(logits)
        prob, pred = torch.max(logits, dim=-1)

        pred = pred.tolist()
        prob = prob.tolist()
        outp = self.post_process(space_id, pred[0], prob[0])
        outp = decode(x, outp).strip()
        return outp

    def post_process(self, y, pred, prob):
        outp = list()
        for y_, pred_, prob_ in zip(y, pred, prob):
            if prob_ < self.threshold:
                outp.append(y_)
            else:
                if y_ == 1:
                    outp.append(y_)
                else:
                    outp.append(pred_)
        return outp

    def train(self,
              sents: list,
              batch_size: int,
              num_epochs: int,
              lr: float,
              save_path: str,
              model_prefix: str,
              **kwargs):
        self.model.train()
        self.max_len = kwargs['max_len']

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        dataset = SpacingDataset(self.tok, sents, kwargs['max_len'])
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=kwargs['num_workers'])

        best_loss = 1e5

        for epoch in range(num_epochs):
            total_loss = 0
            for inputs, space_id, target in tqdm(dataloader, desc='batch progress'):
                # Remember PyTorch accumulates gradients; zero them out
                self.model.zero_grad()

                inputs = inputs.to(self.device)
                space_id = space_id.to(self.device)
                target = target.to(self.device)

                logits = self.model(inputs, space_id)
                # loss = F.cross_entropy(logits, target, ignore_index=kwargs['ignore_index'])
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.reshape(-1),
                                       ignore_index=dataset.ignore_index)

                # backpropagation
                loss.backward()
                # update the parameters
                optimizer.step()
                total_loss += loss.item()

            if total_loss <= best_loss:
                best_loss = total_loss
                self.save_dict(save_path=save_path, model_prefix=model_prefix)
            print("| Epochs: {} | Training loss: {} |".format(epoch + 1,
                                                              round(total_loss, 4)))

    def load_model(self, model_path: str):
        with open(model_path, 'rb') as modelFile:
            model_dict = dill.load(modelFile)
        model_conf = model_dict['model_conf']
        self.model = TransformerSpaceCorrector(**model_conf)
        try:
            self.model.load_state_dict(model_dict["model_params"])
        except:
            new_dict = OrderedDict()
            for key in model_dict["model_params"].keys():
                new_dict[key.replace('module.', '')] = model_dict["model_params"][key]
            self.model.load_state_dict(new_dict)

        self.model.to(self.device)
        self.model.eval()

    def save_dict(self, save_path, model_prefix):
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, model_prefix+'.modeldict')

        outp_dict = {
            'max_len': self.max_len,
            'model_params': self.model.cpu().state_dict(),
            'model_conf': self.model_conf,
            'model_type': 'pytorch',
        }

        with open(filename, "wb") as file:
            dill.dump(outp_dict, file, protocol=dill.HIGHEST_PROTOCOL)
        self.model.to(self.device)




#
# if __name__ == '__main__':
#     sent = '이런개새끼야'
#     model = SISCoModel()
#     out = model.correct(sent)
#     print(out)
#
