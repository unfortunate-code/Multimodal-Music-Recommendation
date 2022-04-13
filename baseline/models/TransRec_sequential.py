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

# https://github.com/slientGe/Sequential_Recommendation_Tensorflow

class TransRec_sequential(torch.nn.Module):
    def __init__(self, user_train, user_valid, user_num, item_num, emb_dim, maxlen,
                 num_epochs, eval_every, early_stop_trial, learning_rate, reg_lambda, batch_size, device, top_k = 20):
        super().__init__()

        self.user_train = user_train
        self.user_valid = user_valid
        self.user_num = user_num
        self.item_num = item_num
        self.emb_dim = emb_dim
        self.maxlen = maxlen
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
        # user emb
        self.user_emb = nn.Embedding(self.user_num, self.emb_dim)
        # item emb
        self.item_emb = nn.Embedding(self.item_num, self.emb_dim)

        # beta item
        self.Beta = nn.Parameter(torch.zeros(self.item_num, 1))
        # time
        self.T = nn.Parameter(torch.zeros(1, self.emb_dim))

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.reg_lambda)

        # to device
        self.to(self.device)

    def forward(self, user_id, last_item, pos_item=None, neg_item=None, item_indices=None):
        
        # user emb
        tu = self.user_emb(user_id)
        # last item emb
        last_item_emb = self.item_emb(last_item)

        # bound to unit ball
        last_l2norm = torch.clamp(torch.linalg.norm(last_item_emb, dim=-1, keepdim=True), min=1)
        last_item_emb = last_item_emb / last_l2norm

        #  user, last item --> current vector
        output = tu + self.T + last_item_emb

        if pos_item is not None:
            # next item emb
            pos_emb = self.item_emb(pos_item)
            pos_l2norm = torch.clamp(torch.linalg.norm(pos_emb, dim=-1, keepdim=True), min=1)
            pos_emb = pos_emb / pos_l2norm
            # item bias
            bias_pos = self.Beta[pos_item]
            # final score
            pos_score = bias_pos - torch.dist(output, pos_emb, p=2)

            neg_emb = self.item_emb(neg_item)
            neg_l2norm = torch.clamp(torch.linalg.norm(neg_emb, dim=-1, keepdim=True), min=1)
            neg_emb = neg_emb / neg_l2norm
            bias_neg = self.Beta[neg_item]
            neg_score = bias_neg - torch.dist(output, neg_emb, p=2)

            return pos_score, neg_score

        if item_indices is not None:
            item_embs = self.item_emb(item_indices)  # (U, I, C)
            item_l2norm = torch.clamp(torch.linalg.norm(item_embs, dim=-1, keepdim=True), min=1)
            item_embs = item_embs / item_l2norm

            bias_item = self.Beta[item_indices]

            score = bias_item - torch.dist(output, item_embs, p=2)                       

            return score

    def fit(self):
        train_loader = DataLoader(list(self.user_train.keys()), batch_size=self.batch_size, shuffle=True)

        top_k = self.top_k
        best_recall = 0
        num_trials = 0
        for epoch in range(1, self.num_epochs+1):
            # Train
            self.train()
            for b, batch_idxes in enumerate(tqdm(train_loader, total=len(train_loader))):
                batch_user = []
                batch_last_item = []
                batch_next_item = []
                batch_neg_item = []
                for i, batch_idx in enumerate(batch_idxes):
                    u = int(batch_idx)
                    seq = self.user_train[u]
                    batch_user += [u] * (len(seq)-1)
                    batch_last_item += seq[:-1]
                    batch_next_item += seq[1:]
                    batch_neg_item += [random_negitem(0, self.item_num, seq) for _ in range(len(seq)-1)]

                loss = self.train_model_per_batch(batch_user, batch_last_item, batch_next_item, batch_neg_item)

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

                print(f'epoch {epoch} train_loss = {loss:.8f} valid_recall@{top_k} = {recall:.4f} valid_ndcg@{top_k} = {ndcg:.4f}')
            else:
                print(f'epoch {epoch} train_loss = {loss:.4f}')
        return

    def train_model_per_batch(self, u, last_item, pos_seqs, neg_seqs):
        # to tensor

        u = torch.LongTensor(u).to(self.device)
        last_item = torch.LongTensor(last_item).to(self.device)
        pos_seqs = torch.LongTensor(pos_seqs).to(self.device)
        neg_seqs = torch.LongTensor(neg_seqs).to(self.device)

        # grad initialization
        self.optimizer.zero_grad()

        # model forwrad
        pos_logits, neg_logits = self.forward(u, last_item, pos_item=pos_seqs, neg_item=neg_seqs)

        # loss
        loss = -1 * torch.mean(torch.log(torch.sigmoid(torch.squeeze(pos_logits - neg_logits))), dim=-1)
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
            user = torch.LongTensor(users).to(self.device)
            last_item = torch.LongTensor(log_seqs[:, -1]).to(self.device)
            item_indices = torch.LongTensor(item_indices).to(self.device)
            logits = self.forward(user, last_item, item_indices=item_indices)
        return logits.squeeze()

    def restore(self):
        with open(f"saves/{self.__class__.__name__}_best_model.pt", 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)



# sampler for batch generation


def random_negitem(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t
