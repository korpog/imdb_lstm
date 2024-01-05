import re
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from spacy.lang.en.stop_words import STOP_WORDS

device = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv('data/IMDB Dataset.csv')

X, y = df['review'].values, df['sentiment'].values
x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y)


def preprocess_string(s):
    # remove non-word characters, whitespaces and digits
    re.sub(r"[^\w\s]", '', s)
    re.sub(r"[\s+]", '', s)
    re.sub(r"[\d]", '', s)

    return s


def tokenize(x_train):
    word_list = []

    for text in x_train:
        for word in text.lower().split():
            word = preprocess_string(word)
            if word not in STOP_WORDS and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:2000]
    vocab = {w: i+1 for i, w in enumerate(corpus_)}

    return vocab

vocabulary = tokenize(x_train)

with open('vocabulary.pickle', 'wb') as file:
    pickle.dump(vocabulary, file)