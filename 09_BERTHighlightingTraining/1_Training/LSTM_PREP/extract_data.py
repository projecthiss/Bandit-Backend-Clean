import pickle
import numpy as np
import os
import my_data
import lstm_prep

import spacy

nlp = spacy.load("en_core_web_lg")

data = my_data.get_data()
print("gotData")
X_train = []
Y_train = []

words = []
tags = []

with_slash = False
n_omitted = 0
a=0
for d in data:
    print(a)
    a+=1

    tok, tag = lstm_prep.document_to_LSTM_example(d)
    for i in range(len(tok)):
        words.append(tok[i].lower())
        tags.append(tag[i])

    X_train.append(tok)
    Y_train.append(tag)
del a
print('OMITTED sentences: ', n_omitted, '\n')
print('TOTAL NO OF SAMPLES: ', len(X_train), '\n')

print('sample X_train: ', X_train[1], '\n')
print('sample Y_train: ', Y_train[1], '\n')
i = 0

with open("all_highlighted_tickets.txt", encoding="utf8") as file:
    i = 0
    for line in file.readlines():
        i = i + 1

        if i % 1000 == 0:
            print(i)
        nl = nlp(line)
        for tok in nl:
            words.append(tok.lemma_.lower())

words = set(words)
tags = set(tags)

print('VOCAB SIZE: ', len(words))
print('TOTAL TAGS: ', len(tags))

assert len(X_train) == len(Y_train)

word2int = {}
int2word = {}

for i, word in enumerate(words):
    word2int[word] = i + 1
    int2word[i + 1] = word

tag2int = {"irrelevant": 0, "system": 1, "source": 2, "faultdescription": 3, "servicerequest": 4, "other":5}
int2tag = {0: "irrelevant", 1: "system", 2: "source", 3: "faultdescription", 4: "servicerequest", 5:"other"}

#for i, tag in enumerate(tags):
#    tag2int[tag] = i
#    int2tag[i] = tag

X_train_numberised = []
Y_train_numberised = []

for sentence in X_train:
    tempX = []
    for word in sentence:
        tempX.append(word2int[word])
    X_train_numberised.append(tempX)

for tags in Y_train:
    tempY = []
    for tag in tags:
        tempY.append(tag2int[tag])
    Y_train_numberised.append(tempY)

print('sample X_train_numberised: ', X_train_numberised[1], '\n')
print('sample Y_train_numberised: ', Y_train_numberised[1], '\n')

X_train_numberised = np.asarray(X_train_numberised)
Y_train_numberised = np.asarray(Y_train_numberised)

pickle_files = [X_train_numberised, Y_train_numberised, word2int, int2word, tag2int, int2tag]

if not os.path.exists('PickledData/'):
    print('MAKING DIRECTORY PickledData/ to save pickled glove file')
    os.makedirs('PickledData/')

with open('PickledData/data.pkl', 'wb') as f:
    pickle.dump(pickle_files, f)

print('Saved as pickle file')
