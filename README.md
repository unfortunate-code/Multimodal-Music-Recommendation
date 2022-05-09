# Multimodal-Music-Recommendation
Recommending music with a multimodal approach
# Work partition
Jae Young Kim Collected different versions
of playlist data and preprocess. Developed and
ran baseline experiments (TransRec, SASRec,
Bert4rec). Experimented with ‘Multimodal Fusion
Based Attentive Networks for Sequential Music
Recommendation’ Paper. Modified BERT4Rec to
work for multimodal setup. Experimented with rec-
ommendation system under various configurations.
Phakawat Wangkriangkri Collected audio data
using Spotify API and preprocessed them. Created
mappings between audio and genres. Generated
audio embeddings using VAE under different con-
figurations. Setup CARC job script experiment
environment for teammates. Experimented with
recommendation system under various configura-
tions.
Anirudh Alameluvari Collected lyrics and al-
bum data and preprocessed both. Generated map-
ping between datasets for intersection. Generated
embeddings for lyrics using 2 word embedding
techniques and 2 sentence transformer techniques
(applied PCA to get the embeddings to 128 dimen-
sions). Collected genres data (Not used as intersec-
tion was very low). Experimented with recommen-
dation system under various configurations.
Jessica D’Souza Collected artist biography from
MSD-A and scrapped missing artist biography
from wikipedia. Generated mapping between artist
bio and playlist data. Generated artist embeddings
using different techniques like SBERT + PCA,
BERT fine tuned on genre classification. Modified
BERT4Rec to work for multimodal setup. Experi-
mented with recommendation system under various
configurations.
Xiaoying Zhang Collected tweet playlist data
and preprocess. Collected audio data using Spotify
API and preprocessed them. Generated album art
embedding using AE and VAE under different con-
figurations. Recovered embedding that miss data
using audio embedding. Experimented with recom-
mendation system under various configurations.
