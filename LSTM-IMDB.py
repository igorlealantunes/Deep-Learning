'''Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
# Notes
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM
from keras.datasets import imdb
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import functools

max_features = 45000
INDEX_FROM = 3
maxlen = 600
batch_size = 64

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features, index_from=INDEX_FROM)

print("x", x_train)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')

# Puts zeros in short reviews and cut the long ones
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()

# max_features = number of distinct words in the input
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())



print('Train...')
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=6,
          validation_data=(x_test, y_test))

#model.save("sentiment-model-45000-600.hr")

#print("Loading pre treined model...")

model = load_model("sentiment-model-45000-600.hr")


#print("evaluating model...")
#score, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)


# load my own dataset
data = pd.read_csv('~/Desktop/news.csv')

### Pre process the text


"""
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)

X = tokenizer.texts_to_sequences(data['text'].values)
X = sequence.pad_sequences(X, maxlen=maxlen)
"""

word_to_id = imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}

indices = data['id'].values
google_sentiment = data['semtiment'].values

tokenized_text = []

for new in data['text']:

	tokenized_text.append( np.array( [ word_to_id[word] if word in word_to_id and word_to_id[word] < max_features else 0 for word in text_to_word_sequence(new)]))

tokenized_text = np.asarray(tokenized_text)

tokenized_text = sequence.pad_sequences(tokenized_text, maxlen=maxlen)

sentiment = model.predict(tokenized_text, verbose=1)

def conver_back(el):
	return ' '.join(id_to_word[w] for w in el)


"""
#print('Test score:', score)
#print('Test accuracy:', acc)
#
# Resultados nao satisfatorios : 
# 	1. Banco de treinamento com contexto diferente do utilizado
# 	2. Numero de palavras limitado (600) 
# 	3. Texto com pouca (ou nenhuma) tendencia a opiniao -- noticias
# 	4. 45k palavras mais comuns, consiguimos reconstruir as noticias quase perfeitamente
"""







