import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from spacy.lang.en.stop_words import STOP_WORDS

df = pd.read_csv('data/IMDB Dataset.csv')

X, y = df['review'].values, df['sentiment'].values
x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y)


def preprocess_string(s):
    # remove non-word characters, whitespaces and digits
    re.sub(r"[^\w\s]", '', s)
    re.sub(r"[\s+]", '', s)
    re.sub(r"[\d]", '', s)

    return s


def tokenize(x_train, x_test, y_train, y_test):
    word_list = []

    for text in x_train:
        for word in text.lower().split():
            word = preprocess_string(word)
            if word not in STOP_WORDS and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    vocab = {w: i+1 for i, w in enumerate(corpus_)}

    final_x_train, final_x_test = [], []
    for text in x_train:
        final_x_train.append([vocab[preprocess_string(word)] for word in text.lower().split()
                              if preprocess_string(word) in vocab.keys()])
    for text in x_test:
        final_x_test.append([vocab[preprocess_string(word)] for word in text.lower().split()
                             if preprocess_string(word) in vocab.keys()])

    label_train = [1 if label == 'positive' else 0 for label in y_train]
    label_test = [1 if label == 'positive' else 0 for label in y_test]

    return final_x_train, final_x_test, label_train, label_test, vocab

def padding(sentences, length):
    features = np.zeros((len(sentences), length), dtype=int)
    for i, review in enumerate(sentences):
         if len(review) != 0:
              features[i, -len(review):] = np.array(review)[:length]
    return features

x_train, x_test, y_train, y_test, vocab = tokenize(x_train, x_test, y_train, y_test)

x_train_padded = padding(x_train, 500)
x_test_padded = padding(x_test, 500)

