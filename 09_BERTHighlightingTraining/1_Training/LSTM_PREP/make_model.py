import numpy as np
import pickle, sys, os

from keras.utils import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from numpy import average

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import random

random.seed(2)
import numpy as np

np.random.seed(2)
import tensorflow as tf

tf.compat.v1.set_random_seed(2)

import spacy

nlp = spacy.load("en_core_web_lg")

# PARAMETERS ================
MAX_SEQUENCE_LENGTH = 5000
EMBEDDING_DIM = 300
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 4

with open('PickledData/data.pkl', 'rb') as f:
    X, y, word2int, int2word, tag2int, int2tag = pickle.load(f)

print(len(X))
X, y = shuffle(X, y)


def generator(all_X, all_y, n_classes, batch_size=BATCH_SIZE):
    num_samples = len(all_X)

    while True:

        for offset in range(0, num_samples, batch_size):
            X = all_X[offset:offset + batch_size]
            y = all_y[offset:offset + batch_size]

            y = to_categorical(y, num_classes=n_classes)

            yield shuffle(X, y)


n_tags = len(tag2int)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

# y = to_categorical(y, num_classes=len(tag2int) + 1)

print('TOTAL TAGS', len(tag2int))
print('TOTAL WORDS', len(word2int))

# shuffle the data
X, y = shuffle(X, y)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=2)

# split training data into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_SPLIT, random_state=1)



with open('all_data_lstm_split.pkl', 'wb') as f:
    data = [X_train, X_val, X_test, y_train, y_val, y_test]
    pickle.dump(data, f)

#with open("all_data_lstm_split.pkl", "rb") as f:
#    X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)

y_train = pad_sequences(y_train, maxlen=MAX_SEQUENCE_LENGTH)
y_val = pad_sequences(y_val, maxlen=MAX_SEQUENCE_LENGTH)
y_test = pad_sequences(y_test, maxlen=MAX_SEQUENCE_LENGTH)

n_train_samples = X_train.shape[0]
n_val_samples = X_val.shape[0]
n_test_samples = X_test.shape[0]

print('We have %d TRAINING samples' % n_train_samples)
print('We have %d VALIDATION samples' % n_val_samples)
print('We have %d TEST samples' % n_test_samples)

# make generators for training and validation
# train_generator = generator(all_X=X_train, all_y=y_train, n_classes=n_tags + 1)
# validation_generator = generator(all_X=X_val, all_y=y_val, n_classes=n_tags + 1)


# print('Total %s word vectors.' % len(embeddings_index))

# + 1 to include the unkown word
embedding_matrix = np.random.random((len(word2int) + 1, EMBEDDING_DIM))

try:
    with open('PickledData/em_matrix.pkl', 'rb') as el:
        embedding_matrix = pickle.load(el)
        embedding_matrix = embedding_matrix[0]

except:
    for word, i in word2int.items():
        embedding_vector = nlp(nlp(word)[0].lemma_).vector
        if i % 1000 == 0:
            print(i)
        if embedding_vector is not None:
            # words not found in embeddings_index will remain unchanged and thus will be random.
            embedding_matrix[i] = embedding_vector
    pickle_embedding_matrix = [embedding_matrix]
    with open('PickledData/em_matrix.pkl', 'wb') as f:
        pickle.dump(pickle_embedding_matrix, f)
    print('Saved as pickle file')

print('Embedding matrix shape', embedding_matrix.shape)
print('X_train shape', X_train.shape)

embedding_layer = Embedding(len(word2int) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

l_lstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(embedded_sequences)
# preds = TimeDistributed(Dense(n_tags, activation='softmax'))(l_lstm)
preds = Dense(n_tags, activation='softmax')(l_lstm)

model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Bidirectional LSTM")
model.summary()

# model.fit_generator(train_generator,
#                    steps_per_epoch=n_train_samples // BATCH_SIZE,
#                    validation_data=validation_generator,
#                    validation_steps=n_val_samples // BATCH_SIZE,
#                    epochs=2,
#                    verbose=1,
#                    workers=1)

y_train = to_categorical(y_train, num_classes=n_tags)
y_val = to_categorical(y_val, num_classes=n_tags)
print("Fitting")
print(len(X_train))
print(len(y_train))
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, verbose=1)

if not os.path.exists('Models/'):
    print('MAKING DIRECTORY Models/ to save model file')
    os.makedirs('Models/')

train = True
#train = False

if train:
    model.save('Models/model.h5')
    print('MODEL SAVED in Models/ as model.h5')
else:
    from keras.models import load_model

    model = load_model('Models/model.h5')

y_test = to_categorical(y_test, num_classes=n_tags)

#test_results = model.evaluate(X_test, y_test, verbose=0)
#print('TEST LOSS %f \nTEST ACCURACY: %f' % (test_results[0], test_results[1]))

new_y_test = []

for example in y_test:
    current_example = []
    for word in example:
        current_example.append(np.argmax(word))
    new_y_test.append(current_example)
y_test = new_y_test

x_pred = model.predict_on_batch(X_test)

x_predicted = []
curEx = 0
new_y_test = []

new_x_pred = []
for example in X_test:
    i = 0
    for word in example:
        if word != 0:
            new_y_test.append(y_test[curEx][i])
            new_x_pred.append(np.argmax(x_pred[curEx][i]))
        i += 1
    curEx += 1
x_predicted = new_x_pred
y_test = new_y_test
a = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
b = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
for i in x_predicted:
    a[i] += 1
for i in y_test:
    b[i] += 1
print(a)
print(b)

print("Bi-LSTM Accuracy Score -> ", accuracy_score(y_test, x_predicted))

print("Bi-LSTM Precision Score (AVERAGE_NONE)-> ", precision_score(y_test, x_predicted, average=None))
print("Bi-LSTM Precision Score (AVERAGE_micro)-> ", precision_score(y_test, x_predicted, average='micro'))
print("Bi-LSTM Precision Score (AVERAGE_macro)-> ", precision_score(y_test, x_predicted, average='macro'))
print("Bi-LSTM Precision Score (AVERAGE_weighted)-> ", precision_score(y_test, x_predicted, average='weighted'))

print("Bi-LSTM Recall Score(AVERAGE_NONE) -> ", recall_score(y_test, x_predicted, average=None))
print("Bi-LSTM Recall Score(AVERAGE_micro) -> ", recall_score(y_test, x_predicted, average='micro'))
print("Bi-LSTM Recall Score(AVERAGE_macro) -> ", recall_score(y_test, x_predicted, average='macro'))
print("Bi-LSTM Recall Score(AVERAGE_weighted) -> ", recall_score(y_test, x_predicted, average='weighted'))

print("Bi-LSTM F1 Score (AVERAGE_NONE)-> ", f1_score(y_test, x_predicted, average=None))
print("Bi-LSTM F1 Score (AVERAGE_micro)-> ", f1_score(y_test, x_predicted, average='micro'))
print("Bi-LSTM F1 Score (AVERAGE_macro)-> ", f1_score(y_test, x_predicted, average='macro'))
print("Bi-LSTM F1 Score (AVERAGE_weighted)-> ", f1_score(y_test, x_predicted, average='weighted'))

# print("Bi-LSTM Log_loss -> ", log_loss(y_test, x_predicted))

matrix = confusion_matrix(y_test, x_predicted)
df_cm = pd.DataFrame(matrix, index=[i for i in tag2int],
                     columns=[i for i in tag2int])
# plt.figure(figsize = (10,7))
# sn.heatmap(df_cm, annot=True)
# plt.show()

ax = plt.subplot()
sn.heatmap(df_cm, annot=True, fmt='g', ax=ax);  # annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')

ax.set_title('Confusion Matrix - Bi-LSTM Tagger')
plt.tight_layout()
plt.show()
