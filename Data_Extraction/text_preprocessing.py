from typing import Text
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import glob

TRAIN_PATH = "test_lyrics/train"
VAL_PATH = "test_lyrics/val"
N_WORDS = 300  # the number of words chose from each song


def split_words(dir):
    files = glob.glob(dir+"/*.txt")
    print(files)
    texts = []
    for file in files:
        text = open(file).read()
        sentences = text.split("\n")
        clean_sentences = [s.split(" ") for s in sentences]
        words = [word for s in clean_sentences for word in s]
        lyric = " ".join(words[:N_WORDS])
        texts.append(lyric)
    return texts


def vectorize(texts):
    vec = vectorizer.transform(texts)
    print(vectorizer.get_feature_names_out())
    return vec


vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1.0, analyzer='word', max_features=10000)
texts = split_words(TRAIN_PATH)
vec = vectorizer.fit_transform(texts)
pickle.dump(vectorizer.vocabulary_,open("../Data/feature.pkl","wb"))


def main():
    text = split_words(VAL_PATH)
    train_vec = vectorize(text)
    print(train_vec)

if __name__ == '__main__':
    main()