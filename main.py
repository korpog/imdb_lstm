import re
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


def tokenize(x_train, x_test, y_train, y_test):
    word_list = []

    for text in x_train:
        for word in text.lower().split():
            word = preprocess_string(word)
            if word not in STOP_WORDS and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:2000]
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


x_train, x_test, y_train, y_test, vocab = tokenize(
    x_train, x_test, y_train, y_test)

x_train_padded = padding(x_train, 500)
x_test_padded = padding(x_test, 500)
y_train = np.array(y_train)
y_test = np.array(y_test)

train_data = TensorDataset(torch.from_numpy(
    x_train_padded), torch.from_numpy(y_train))
test_data = TensorDataset(torch.from_numpy(
    x_test_padded), torch.from_numpy(y_test))

batch_size = 50

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

data_iterator = iter(train_loader)
sample_x, sample_y = next(data_iterator)

embedding_dim = 64
output_dim = 1
no_layers = 4
vocab_size = len(vocab) + 1
hidden_dim = 256


class LSTM_NN(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, emdedding_dim, drop_prob=0.5):
        super(LSTM_NN, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.no_layers = no_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=emdedding_dim, hidden_size=self.hidden_dim,
                            num_layers=no_layers, batch_first=True)
        self.droput = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeddings = self.embedding(x)
        lstm_out, hidden = self.lstm(embeddings, hidden)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.droput(lstm_out)
        out = self.fc(out)

        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch

        return sig_out, hidden

    def init_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim
        h0 = torch.zeros((self.no_layers, batch_size,
                         self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers, batch_size,
                         self.hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden


model = LSTM_NN(no_layers, vocab_size, hidden_dim,
                embedding_dim, drop_prob=0.5)

model.to(device)
print(model)

learning_rate = 0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def accuracy(prediction, label):
    pred = torch.round(prediction.squeeze())
    return torch.sum(pred == label.squeeze()).item()


grad_clip = 5
epochs = 5
loss_min = np.Inf

train_losses, valid_losses = [], []
train_accuracy, valid_accuracy = [], []

for epoch in range(epochs):
    tr_losses = []
    tr_acc = 0.0
    model.train()

    hidden = model.init_hidden(batch_size)

    for reviews, labels in train_loader:
        reviews, labels = reviews.to(device), labels.to(device)
        hidden = tuple([e.data for e in hidden])

        model.zero_grad()
        output, hidden = model(reviews, hidden)

        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        tr_losses.append(loss.item())

        acc = accuracy(output, labels)
        tr_acc += acc
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    val_losses = []
    val_acc = 0.0
    model.eval()

    hidden_valid = model.init_hidden(batch_size)

    for reviews, labels in test_loader:
        reviews, labels = reviews.to(device), labels.to(device)
        hidden_valid = tuple([e.data for e in hidden_valid])

        model.zero_grad()
        output, hidden_valid = model(reviews, hidden_valid)

        loss = criterion(output.squeeze(), labels.float())
        val_losses.append(loss.item())

        acc = accuracy(output, labels)
        val_acc += acc

    epoch_train_losses = np.mean(tr_losses)
    epoch_valid_losses = np.mean(val_losses)

    epoch_train_accuracy = tr_acc/len(train_loader.dataset)
    epoch_valid_accuracy = val_acc/len(test_loader.dataset)

    train_losses.append(epoch_train_losses)
    valid_losses.append(epoch_valid_losses)

    train_accuracy.append(epoch_train_accuracy)
    valid_accuracy.append(epoch_valid_accuracy)

    print(f'Epoch: {epoch+1}')
    print(
        f'Training loss: {epoch_train_losses:.3f} \t\t Validation loss: {epoch_valid_losses:.3f}')
    print(
        f'Training accuracy: {epoch_train_accuracy:.3f} \t Validation accuracy: {epoch_valid_accuracy:.3f}')
    print(10*'=')

torch.save(model.state_dict(), 'model_state_dict.pt')
