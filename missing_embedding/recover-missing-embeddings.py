import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# data
AUDIO_EMBEDDING = "npy/individual_audio_embedding.npy"
audio_embeddings = np.load(AUDIO_EMBEDDING, allow_pickle=True)
audio_embeddings = torch.FloatTensor(audio_embeddings)

BATCH_SIZE = 5722
data_load = DataLoader(audio_embeddings,batch_size=BATCH_SIZE,shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('recover_model.p',map_location='cpu'))

output_np = np.zeros(shape=(len(audio_embeddings),128))
for i, chunk in enumerate(data_load):
    output = model(chunk)
    
    print(output.shape)
    np_sub = output.detach().cpu().numpy()
    output_np[i*BATCH_SIZE:i*BATCH_SIZE + len(np_sub)] = np_sub
    print("embedding numpy: ", output_np.shape)

with open('npy/individual_album_embedding.npy', 'wb') as f:
    np.save(f, output_np)

    
