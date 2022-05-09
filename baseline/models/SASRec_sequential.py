from utils import eval_sequential
import os
import math
from time import time
from IPython.terminal.embed import embed
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader
from multiprocessing import Process, Queue

import sys
sys.path.append("..")

# https://github.com/pmixer/SASRec.pytorch


class SASRec_sequential(torch.nn.Module):
    def __init__(self, user_train, user_valid, user_num, item_num, hidden_dim, maxlen, num_blocks, num_heads,
                 num_epochs, eval_every, early_stop_trial, learning_rate, reg_lambda, batch_size, device, top_k = 20):
        super().__init__()

        self.user_train = user_train
        self.user_valid = user_valid
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_dim = hidden_dim
        self.maxlen = maxlen
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.top_k = top_k

        self.num_epochs = num_epochs
        self.eval_every = eval_every
        self.early_stop_trial = early_stop_trial
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.batch_size = batch_size

        self.device = device

        self.build_graph()

    def build_graph(self):
        # item emb
        self.item_emb = torch.nn.Embedding(self.item_num+1, self.hidden_dim, padding_idx=0)
        # pos emb
        self.pos_emb = torch.nn.Embedding(self.maxlen, self.hidden_dim)
        self.emb_dropout = torch.nn.Dropout(p=0.2)

        # Self-Attention Layer
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        # Position-wise Feed-Forward Layer
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.hidden_dim)

        # Self-Attention block
        for _ in range(self.num_blocks):
            # layer normalization layer (self-attention)
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_dim)
            self.attention_layernorms.append(new_attn_layernorm)

            # self-attention layer
            new_attn_layer = torch.nn.MultiheadAttention(self.hidden_dim, self.num_heads, 0.2)
            self.attention_layers.append(new_attn_layer)

            # layer norm layer (position-wise feed-forward)
            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_dim)
            self.forward_layernorms.append(new_fwd_layernorm)

            # position-wise feed-forward layer
            new_fwd_layer = PointWiseFeedForward(self.hidden_dim, 0.2)
            self.forward_layers.append(new_fwd_layer)

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.reg_lambda)

        # to device
        self.to(self.device)

    def forward(self, log_seqs, pos_seqs=None, neg_seqs=None, item_indices=None):
        # item emb(batch, maxlen -> batch, maxlen, hidden_dim)
        seqs = self.item_emb(log_seqs)
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.device))
        seqs = self.emb_dropout(seqs)

        # padding to 0
        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        # masking
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        # Self-attention block
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            # layer normalization
            Q = self.attention_layernorms[i](seqs)
            # self-attention
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            # residual connection
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            # layer normalization
            seqs = self.forward_layernorms[i](seqs)
            # position-wise feed-forward
            seqs += self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        # (batch, maxlen, hidden_dim)
        log_feats = self.last_layernorm(seqs)
        if pos_seqs is not None:
            pos_embs = self.item_emb(pos_seqs)
            pos_logits = (log_feats * pos_embs).sum(dim=-1)
            neg_embs = self.item_emb(neg_seqs)
            neg_logits = (log_feats * neg_embs).sum(dim=-1)
            return pos_logits, neg_logits

        if item_indices is not None:
            # Only use last item vector (batch, hidden_dim)
            final_feat = log_feats[:, -1, :]
            item_embs = self.item_emb(item_indices)
            logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
            return logits

    def fit(self):
        sampler = WarpSampler(self.user_train, self.user_num, self.item_num, batch_size=self.batch_size, maxlen=self.maxlen, n_workers=3)

        top_k = self.top_k
        best_recall = 0
        num_trials = 0
        num_batch = len(self.user_train) // self.batch_size
        for epoch in range(1, self.num_epochs+1):
            # Train
            self.train()
            for step in range(num_batch):
                u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

                loss = self.train_model_per_batch(u, seq, pos, neg)

            # Valid
            if epoch % self.eval_every == 0:
                self.eval()
                ndcg, recall = eval_sequential(self, self.user_train, self.user_valid, None, self.user_num, self.item_num, top_k=top_k, mode='valid')

                if recall > best_recall:
                    best_recall = recall
                    torch.save(self.state_dict(), f"saves/{self.__class__.__name__}_best_model.pt")
                    num_trials = 0
                else:
                    num_trials += 1

                if num_trials >= self.early_stop_trial and self.early_stop_trial > 0:
                    print(f'Early stop at epoch:{epoch}')
                    self.restore()
                    break

                print(f'epoch {epoch} train_loss = {loss:.4f} valid_recall@{top_k} = {recall:.4f} valid_ndcg@{top_k} = {ndcg:.4f}')
            else:
                print(f'epoch {epoch} train_loss = {loss:.4f}')
        return

    def train_model_per_batch(self, u, log_seqs, pos_seqs, neg_seqs):
        # to tensor
        log_seqs = torch.LongTensor(log_seqs).to(self.device)
        pos_seqs = torch.LongTensor(pos_seqs).to(self.device)
        neg_seqs = torch.LongTensor(neg_seqs).to(self.device)

        # grad initialization
        self.optimizer.zero_grad()

        # model forwrad
        pos_logits, neg_logits = self.forward(log_seqs, pos_seqs=pos_seqs, neg_seqs=neg_seqs)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(neg_logits.shape, device=self.device)

        # loss
        indices = torch.where(pos_seqs != 0)
        loss = self.criterion(pos_logits[indices], pos_labels[indices])
        loss += self.criterion(neg_logits[indices], neg_labels[indices])
        for param in self.item_emb.parameters():
            loss += self.reg_lambda * torch.norm(param)

        # backprop
        loss.backward()

        # weight update
        self.optimizer.step()

        return loss

    def predict(self, users, log_seqs, item_indices):
        self.eval()
        with torch.no_grad():
            log_seqs += log_seqs.astype(np.bool).astype(np.long)  # 0 부터 시작 -> 1부터 시작
            log_seqs = torch.LongTensor(log_seqs).to(self.device)
            item_indices += item_indices.astype(np.bool).astype(np.long)  # 0 부터 시작 -> 1부터 시작
            item_indices = torch.LongTensor(item_indices).to(self.device)
            logits = self.forward(log_seqs, item_indices=item_indices)

        return logits[0]

    def restore(self):
        with open(f"saves/{self.__class__.__name__}_best_model.pt", 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


# sampler for batch generation
class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# sampler for batch generation
def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(0, usernum)
        while len(user_train[user]) <= 1:
            user = np.random.randint(0, usernum)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i + 1 
            pos[idx] = nxt + 1
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))

# sampler for batch generation


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t
