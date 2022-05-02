import numpy as np

FILE_ID_IN_ORDER = "embedding/idx2song.data"
EMBEDDING1 = "npy/common_album_embedding.npy"
EMBEDDING2 = "npy/individual_album_embedding.npy"
IDS1 = "npy/common_ids.npy"
IDS2 = "npy/individual_ids.npy"

order_right = np.load(FILE_ID_IN_ORDER, allow_pickle=True)
embeddings_1 = np.load(EMBEDDING1, allow_pickle=True)
embeddings_2 = np.load(EMBEDDING2, allow_pickle=True)
id_1 = np.load(IDS1, allow_pickle=True)
id_2 = np.load(IDS2, allow_pickle=True)

dim = 128
embedding = []

for i in order_right:
    result = np.where(id_1 == i)
    if len(result) > 0 and len(result[0]) > 0:
        embedding.append(embeddings_1[result[0][0]])
    result = np.where(id_2 == i)
    if len(result) > 0 and len(result[0]) > 0:
        embedding.append(embeddings_2[result[0][0]])

embedding = np.reshape(embedding, (-1,dim))
print(embedding.shape)

np.save("npy/album_embedding_recovered",embedding)

