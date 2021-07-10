import os
import pandas as pd
import numpy as np
import nltk
import re
import csv
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from cleaners import *
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential, load_model, model_from_config
import keras.backend as K

current_dir = os.path.dirname(os.path.realpath('model.py'))

X = pd.read_csv(os.path.join(current_dir,'/Data/test.csv'), sep=',', encoding='ISO-8859-1')
#y = X['evaluator_rating']
X = X.dropna(axis=1)
X = X.drop(columns=['Unnamed: 0', 'uniqueId'])

train_essays = X['essay']
print(len(X))
print(len(X['essay']))

sentences = []

for essay in train_essays:
    # Obtaining all sentences from the training essays.
    sentences += essay_to_sentences(essay, remove_stopwords = True)

print(sentences)

# Initializing variables for word2vec model.
num_features = 300 
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)

#model.init_sims(replace=True)

clean_train_essays = []

# Generate training and testing data word vectors.
for essay_v in train_essays:
    clean_train_essays.append(essay_to_wordlist(essay_v, remove_stopwords=True))

print(clean_train_essays)

trainDataVecs = getAvgFeatureVecs(clean_train_essays, model, num_features)

trainDataVecs = np.array(trainDataVecs)

trainDataVecs = np.reshape(trainDataVecs, (trainDataVecs.shape[0], 1, trainDataVecs.shape[1]))

print(trainDataVecs)

lstm_model = load_model(os.path.join(current_dir,"/models/final_lstm5.h5"))

y_pred = lstm_model.predict(trainDataVecs)

liist = []

for i in range(0,len(X)):
    liist.append(np.round(y_pred[i][0],2))

X = pd.read_csv(os.path.join(current_dir,"/Data/test.csv"), sep=',', encoding='ISO-8859-1')

file = open("test_prediction.csv", "w")
writer = csv.writer(file)

for w in range(0,len(X)):

    writer.writerow([X['promptId'][w],X['uniqueId'][w],X['essay'][w], liist[w]])

file.close()
