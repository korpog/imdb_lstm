import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/IMDB Dataset.csv')
print(df.head())

X, y = df['review'].values, df['sentiment'].values
x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y)
