"""
Embarrassingly shallow autoencoders for sparse data, 
Harald Steck,
Arxiv.
"""
import os
import math
import numpy as np

class EASE_implicit():
    def __init__(self, train, valid, reg_lambda):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.reg_lambda = reg_lambda

    def fit(self):   
        train_matrix = self.train
        G = train_matrix.T @ train_matrix
        diag = np.diag_indices(G.shape[0]) # diagonal element index change
        G[diag] += self.reg_lambda # add normalization
        P = np.linalg.inv(G) # P_hat matrix
        
        # P_hat --> W
        self.enc_w = P / (-np.diag(P)) 
        self.enc_w[diag] = 0

        # user-item matrix * W --> prediction matrix
        self.reconstructed = self.train @ self.enc_w

    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]


