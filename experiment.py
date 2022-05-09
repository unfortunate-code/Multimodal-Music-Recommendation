#!/usr/bin/env python
# coding: utf-8

# In[16]:


# import os
# import warnings
import random
# from os.path import join
import torch
import numpy as np
import pandas as pd
# from scipy import sparse
# import torchvision
import pickle


# In[17]:


torch.__version__


# In[18]:


# torchvision.__version__


# In[19]:


import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)


# In[25]:




# import warnings
# import random
# from os.path import join
# import torch
# import numpy as np
# warnings.filterwarnings('ignore')

# set seed
def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

seed = 1
seed_everything(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
top_k = 20


# In[21]:


import pickle
with open('data/playlist.data', 'rb') as f:
    playlists = pickle.load(f)


# In[22]:



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


# In[23]:


top_k = 500


# In[ ]:


# data_dir = 'data/embeddings/'
# embedding = np.load(data_dir + 'lyrics_glove_128.npy')
# embedding = np.append(np.array([np.random.normal(0, 1, 128)]), embedding, axis = 0)
# embedding = np.append(embedding, np.array([np.random.normal(0, 1, 128)]), axis = 0)
# embedding = torch.FloatTensor(embedding)
# print(embedding.size())


# In[ ]:


##########################################################################################################################
# data_dir = 'data/embeddings/'
# with open(data_dir + 'audio_embeddings.p', 'rb') as f:
#     embedding = pickle.load(f)
#     embedding = np.append(np.array([np.random.normal(0, 1, 150)]), embedding, axis = 0)
#     embedding = np.append(embedding, np.array([np.random.normal(0, 1, 150)]), axis = 0)
#     embedding = torch.FloatTensor(embedding)
#     print(embedding.size())
# ############################################################################################################################


# In[31]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
"""
BERT4Rec
"""
from models.Bert4rec_custom import BERTRec_sequential
# from models.BERTRec_sequential import BERTRec_sequential
seed_everything(seed)
#####################################################################################################################
'''
Change hidden to the corresponding embedding dimension
hidden % head should be 0
'''
BERTRec = BERTRec_sequential(user_train_music, user_valid_music, user_num=num_users, item_num=num_items, hidden=300, maxlen=50, n_layers=2, heads=8, mask_prob=0.1,
                                num_epochs=100, eval_every=1, early_stop_trial=5, learning_rate=0.001, reg_lambda=0.0, batch_size=256, device=device, top_k = top_k)
#####################################################################################################################


# In[32]:


BERTRec.fit()




