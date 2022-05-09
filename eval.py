from utils import eval_sequential
from models.Bert4rec_custom import BERTRec_sequential
import torch
import pickle

print('loading data...')
with open('data/playlist.data', 'rb') as f:
    playlists = pickle.load(f)
    
num_users = len(playlists)
num_items = 70229

print(f"# of users: {num_users},  # of items: {num_items}")

music_data = dict()
for idx, i in enumerate(playlists):
    music_data[idx] = i
    
    
user_train_music = dict()
user_valid_music = dict()
user_test_music = dict()

for u in music_data:
    user_valid_music[u] = [music_data[u][-2]]
    user_test_music[u] = [music_data[u][-1]]
    user_train_music[u] = music_data[u][:-2]

print('loading model...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
top_k = 500
BERTRec = BERTRec_sequential(user_train_music, user_valid_music, user_num=num_users, item_num=num_items, hidden=300, maxlen=50, n_layers=2, heads=6, mask_prob=0.1, num_epochs=1, eval_every=1, early_stop_trial=5, learning_rate=0.001, reg_lambda=0.0, batch_size=128, device=device, top_k = top_k)

BERTRec.load_state_dict(torch.load('saves/BERTRec_sequential_best_model.pt'))
BERTRec.eval()

print('eval...')
BERTRec_ndcg_500, BERTRec_recall_500, BERTRec_ndcg_20, BERTRec_recall_20 = eval_sequential(BERTRec, user_train_music, user_valid_music, user_test_music, num_users, num_items, top_k, mode='test')
print(f"\n[BERTRec]\t Test_Recall@500 = {BERTRec_recall_500:.8f} Test_NDCG@500 = {BERTRec_ndcg_500:.8f}")
print(f"\n[BERTRec]\t Test_Recall@20 = {BERTRec_recall_20:.8f} Test_NDCG@20 = {BERTRec_ndcg_20:.8f}")
print("parmas :", sum(p.numel() for p in BERTRec.parameters() if p.requires_grad))
