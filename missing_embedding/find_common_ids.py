import numpy as np

FILE_ID_IN_ORDER = "embedding/idx2song.data"
FILE_MYID = "npy/ids-processed.npy"
ALBUM_EMBEDDING = "ebd/album_embeddings_vae.npy"
AUDIO_EMBEDDING = "ebd/audio_embeddings.p"

order_right = np.load(FILE_ID_IN_ORDER, allow_pickle=True)
order_my = np.load(FILE_MYID, allow_pickle=True)
album_embeddings = np.load(ALBUM_EMBEDDING, allow_pickle=True)
audio_embeddings = np.load(AUDIO_EMBEDDING, allow_pickle=True)

dim = 128
common_album_embedding = []
common_audio_embedding = []
individual_audio_embedding = []
individual_ids = []
common_ids = []

for i in order_right:
    result = np.where(order_my == order_right[i])
    if len(result) > 0 and len(result[0]) > 0:
        common_album_embedding.append(album_embeddings[i])
        common_audio_embedding.append(audio_embeddings[i])
        common_ids.append(i)
    else:
        individual_audio_embedding.append(audio_embeddings[i])
        individual_ids.append(i)


common_album_embedding = np.reshape(common_album_embedding, (-1,dim))
common_audio_embedding = np.reshape(common_audio_embedding, (-1,dim))
individual_audio_embedding = np.reshape(individual_audio_embedding, (-1,dim))

print(common_album_embedding.shape)
print(common_audio_embedding.shape)
print(individual_audio_embedding.shape)
print(len(common_ids))
print(len(individual_ids))

np.save("npy/common_album_embedding",common_album_embedding)
np.save("npy/common_audio_embedding",common_audio_embedding)
np.save("npy/individual_audio_embedding",individual_audio_embedding)
np.save("npy/common_ids",np.array(common_ids))
np.save("npy/individual_ids",np.array(individual_ids))

