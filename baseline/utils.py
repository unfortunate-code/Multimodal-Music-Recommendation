import os
import math
import ast
import collections
import random
import numpy as np
import pandas as pd
import pickle
import copy
import torch
import dgl

from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss
from google_drive_downloader import GoogleDriveDownloader as gdd
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from metrics import calc_metrics_at_k
# from IPython import embed


# 2d array to dictionary
# input: [[user_id, item_id, timestamp], ...]  numpy array
# output: {user_id: [item1, item2, ......], ...} dictionary
def make_to_dict(data):
    data_dict = {}
    cur_user = -1
    tmp_user = []
    for row in data:
        user_id, item_id = row[0], row[1]
        if user_id != cur_user:
            if cur_user != -1:
                tmp = np.asarray(tmp_user)
                tmp_items = tmp[:, 1]
                data_dict[cur_user] = list(tmp_items)

            cur_user = user_id
            tmp_user = []
        tmp_user.append(row)

    if cur_user != -1:
        tmp = np.asarray(tmp_user)
        tmp_items = tmp[:, 1]
        data_dict[cur_user] = list(tmp_items)

    return data_dict



def compute_metrics(pred_u, target_u, top_k):
    pred_k = pred_u[:top_k]
    num_target_items = len(target_u)

    hits_k = [(i + 1, item) for i, item in enumerate(pred_k) if item in target_u]
    num_hits = len(hits_k)

    idcg_k = 0.0
    for i in range(1, min(num_target_items, top_k) + 1):
        idcg_k += 1 / math.log(i + 1, 2)

    dcg_k = 0.0
    for idx, item in hits_k:
        dcg_k += 1 / math.log(idx + 1, 2)

    prec_k = num_hits / top_k
    recall_k = num_hits / min(num_target_items, top_k)
    ndcg_k = dcg_k / idcg_k

    return prec_k, recall_k, ndcg_k


def eval_implicit(model, train_data, test_data, top_k):
    prec_list = []
    recall_list = []
    ndcg_list = []

    if 'Item' in model.__class__.__name__:
        num_users, num_items = train_data.shape
        pred_matrix = np.zeros((num_users, num_items))

        for item_id in range(len(train_data.T)):
            train_by_item = train_data[:, item_id]
            missing_user_ids = np.where(train_by_item == 0)[0]  # missing user_id

            pred_u_score = model.predict(item_id, missing_user_ids)
            pred_matrix[missing_user_ids, item_id] = pred_u_score

        for user_id in range(len(train_data)):
            train_by_user = train_data[user_id]
            missing_item_ids = np.where(train_by_user == 0)[0]  # missing item_id

            pred_u_score = pred_matrix[user_id, missing_item_ids]
            pred_u_idx = np.argsort(pred_u_score)[::-1]
            pred_u = missing_item_ids[pred_u_idx]

            test_by_user = test_data[user_id]
            target_u = np.where(test_by_user >= 0.5)[0]

            prec_k, recall_k, ndcg_k = compute_metrics(pred_u, target_u, top_k)
            prec_list.append(prec_k)
            recall_list.append(recall_k)
            ndcg_list.append(ndcg_k)
    else:
        for user_id in range(len(train_data)):
            train_by_user = train_data[user_id]
            missing_item_ids = np.where(train_by_user == 0)[0]  # missing item_id

            pred_u_score = model.predict(user_id, missing_item_ids)
            pred_u_idx = np.argsort(pred_u_score)[::-1]  # 내림차순 정렬
            pred_u = missing_item_ids[pred_u_idx]

            test_by_user = test_data[user_id]
            target_u = np.where(test_by_user >= 0.5)[0]

            prec_k, recall_k, ndcg_k = compute_metrics(pred_u, target_u, top_k)
            prec_list.append(prec_k)
            recall_list.append(recall_k)
            ndcg_list.append(ndcg_k)

    return np.mean(prec_list), np.mean(recall_list), np.mean(ndcg_list)



def eval_sequential(model, train_data, valid_data, test_data, usernum, itemnum, top_k=100, mode='valid'):
    
    import random

    if test_data == None:
        keys = random.sample(list(train_data.keys()),1000)
        new_train = dict()
        new_valid = dict()
        new_test = dict()
        for i in keys:
            new_train[i] = train_data[i]
            new_valid[i] = valid_data[i]
            if test_data != None:
                new_test[i] = test_data[i]
        if test_data == None:
            new_test = None
    else:
        keys = range(usernum)
        new_train = train_data
        new_valid = valid_data
        new_test = test_data
    
    
    [train_data, valid_data, test_data, usernum, itemnum] = copy.deepcopy([new_train, new_valid, new_test, usernum, itemnum])

    NDCG = 0.0
    eval_user = 0.0
    HT = 0.0
    users = range(len(keys))
    for u in tqdm(keys, desc=f'{mode}', dynamic_ncols=True):
        if len(train_data[u]) < 1 or len(train_data[u]) < 1: continue

        seq = np.zeros([model.maxlen], dtype=np.int32)
        idx = model.maxlen - 1
        if mode == 'test':
            seq[idx] = valid_data[u][0]

        for i in reversed(train_data[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = train_data[u]
        target_item = test_data[u][0] if mode == 'test' else valid_data[u][0]
        if mode == 'test':
            rated.append(valid_data[u][0])
        item_idx = list(range(itemnum))

        predictions = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions[rated] = -np.inf

        sorted_items = np.argsort(-predictions.cpu().detach().numpy())

        rank = np.where(sorted_items == target_item)[0][0]
        eval_user += 1

        if rank < top_k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / eval_user, HT / eval_user


