import bert_prep
import my_data
import spacy
import tensorflow as tf
import pickle
import os

import random

random.seed(2)
import numpy as np

np.random.seed(2)
tf.compat.v1.set_random_seed(2)

# PARAMETERS ================
MAX_SEQUENCE_LENGTH = 2500
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 4

BERT_MODEL = "bert-base-uncased"

nlp = spacy.load("en_core_web_lg")

data = my_data.get_data()

label_names = ["irrelevant", "system", "source", "faultdescription", "servicerequest", "other"]
label_tags = [0, 1, 2, 3, 4, 5]


def label_names_to_tags(array):
    new_arr = []
    for a in array:
        b = 0
        while label_names[b] != a:
            b += 1
        new_arr.append(b)
    return new_arr


try:
    with open('all_words.pkl', 'rb') as el:
        words = pickle.load(el)
except:
    words = []
    with open("all_words.txt", encoding="utf8") as file:
        i = 0
        for line in file.readlines():
            i = i + 1
            if i % 1000 == 0:
                print(i)
            nl = nlp(line)
            for tok in nl:
                words.append(tok.text.lower())
                words.append(tok.lemma_.lower())
    words = set(words)
    with open('all_words.pkl', 'wb') as f:
        pickle.dump(words, f)

example_ner_tags = []
example_tokens = []

from datasets import load_dataset

raw_datasets = ""
data_files = {"train": "my_dataframe_train.pkl",
              "validation": "my_dataframe_validation.pkl", "test": "my_dataframe_test.pkl"}
try:
    raw_datasets = load_dataset("pandas", data_files=data_files)
except:
    print("hi")
    id_count = 0

    # for d in data:
    #    tok, tag = bert_prep.document_to_LSTM_example(d)
    #    tag = label_names_to_tags(tag)
    #    example_ner_tags.append(tag)
    #    example_tokens.append(tok)
    local_data = ""
    with open("all_data_lstm_split.pkl", "rb") as f:
        X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)

    with open('data.pkl', 'rb') as f:
        X, y, word2int, int2word, tag2int, int2tag = pickle.load(f)
    # print(int2word)
    # print(word2int)
    new_X_train = []
    for a in X_train:
        loc_x = []
        for z in a:
            if z != 0:
                loc_x.append(int2word[z])
        new_X_train.append(loc_x)
    X_train = new_X_train

    new_X_test = []
    for a in X_test:
        loc_x = []
        for z in a:
            if z != 0:
                loc_x.append(int2word[z])
        new_X_test.append(loc_x)
    X_test = new_X_test

    new_X_vali = []
    for a in X_val:
        loc_x = []
        for z in a:
            if z != 0:
                loc_x.append(int2word[z])
        new_X_vali.append(loc_x)
    X_val = new_X_vali

    new_y_train = []
    for a in y_train:
        loc_y = []
        for z in a:
            loc_y.append(int2tag[z])
        new_y_train.append(loc_y)
    y_train = new_y_train

    new_y_val = []
    for a in y_val:
        loc_y = []
        for z in a:
            loc_y.append(int2tag[z])
        new_y_val.append(loc_y)
    y_val = new_y_val

    new_y_test = []
    for a in y_test:
        loc_y = []
        for z in a:
            loc_y.append(int2tag[z])
        new_y_test.append(loc_y)
    y_test = new_y_test

    new_y_train = []
    for a in y_train:
        loc_y = label_names_to_tags(a)
        new_y_train.append(loc_y)
    y_train = new_y_train

    new_y_val = []
    for a in y_val:
        loc_y = label_names_to_tags(a)
        new_y_val.append(loc_y)
    y_val = new_y_val

    new_y_test = []
    for a in y_test:
        loc_y = label_names_to_tags(a)
        new_y_test.append(loc_y)
    y_test = new_y_test

    import pandas as pd

    train = pd.DataFrame({"ner_tags": y_train, "tokens": X_train})
    test = pd.DataFrame({"ner_tags": y_test, "tokens": X_test})
    validation = pd.DataFrame({"ner_tags": y_val, "tokens": X_val})

    with open('my_dataframe_train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open('my_dataframe_test.pkl', 'wb') as f:
        pickle.dump(test, f)
    with open('my_dataframe_validation.pkl', 'wb') as f:
        pickle.dump(validation, f)

    raw_datasets = load_dataset("pandas", data_files=data_files)

a = raw_datasets["train"][0]["ner_tags"]
b = raw_datasets["train"][0]["tokens"]
print(a)
print(b)

ner_feature = raw_datasets["train"].features["ner_tags"]
print(ner_feature)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
print(inputs.tokens())
print(inputs.word_ids())

print("----------------------")


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]

            new_labels.append(label)

    return new_labels


labels = raw_datasets["train"][0]["ner_tags"]
word_ids = inputs.word_ids()
print(labels)
print(align_labels_with_tokens(labels, word_ids))

print("----------------------")


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer, return_tensors="tf"
)

tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=1,
)
tf_vali_dataset = tokenized_datasets["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=1,
)

tf_eval_dataset = tokenized_datasets["test"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=1,
)

id2label = {str(i): label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}
print(label2id)
print(id2label)
print("----------------------")


from transformers import TFAutoModelForTokenClassification

# learn or test learned model
if (False):
    model = TFAutoModelForTokenClassification.from_pretrained(
        BERT_MODEL,
        id2label=id2label,
        label2id=label2id,
    )
    print(model.config.num_labels)

    print("----------------------")

    from transformers import create_optimizer
    import tensorflow as tf

    # Train in mixed-precision float16
    # Comment this line out if you're using a GPU that will not benefit from this
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
    # by the total number of epochs. Note that the tf_train_dataset here is a batched tf.data.Dataset,
    # not the original Hugging Face Dataset, so its len() is already num_samples // batch_size.
    num_epochs = 2
    num_train_steps = len(tf_train_dataset) * num_epochs

    optimizer, schedule = create_optimizer(
        init_lr=2e-5,
        num_warmup_steps=0,
        num_train_steps=num_train_steps,
        weight_decay_rate=0.01,
    )
    model.compile(optimizer=optimizer)
    train = True

    print(len(tf_train_dataset))
    print(len(tf_vali_dataset))
    print(len(tf_eval_dataset))
    model.fit(
        tf_train_dataset,
        validation_data=tf_vali_dataset,
        epochs=num_epochs,
        verbose=1
    )

    model.save_pretrained("trained_bert")

else:
    model = TFAutoModelForTokenClassification.from_pretrained("trained_bert", local_files_only=True)
    print("loaded")

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

all_predictions = []
all_labels = []
print (tf_eval_dataset)
for batch in tf_eval_dataset:
    labels = batch["labels"]
    outputs = model(**batch)
    logits = outputs.logits
    predicted_label_classes = np.argmax(logits, axis=-1)
    #for id in predicted_label_classes.squeeze().tolist():
    #    print(id)
    predicted_labels = [model.config.id2label[id] for id in predicted_label_classes.squeeze().tolist()]

    for id, label, true_label in zip(batch["input_ids"][0], predicted_labels, labels[0]):
        if (not (tokenizer.decode(id)[0:2] == "##" or
                 (tokenizer.decode(id)[0] == "[" and tokenizer.decode(id)[-1] == "]"))):
            all_predictions.append(label)
            all_labels.append(label_names[true_label])
            #print(tokenizer.decode(id), label,label_names[true_label])

    # logits = model.predict(batch)["logits"]
    # labels = batch["labels"]
    # predictions = np.argmax(logits, axis=-1)
    # print(predictions)
    #
    # predicted_labels = [model.config.id2label[id] for id in predictions.squeeze().tolist()]
    # print(predicted_labels)
    # print("HI")
    # print(batch)
    # for id, label in zip(batch["input_ids"].squeeze().tolist(), predicted_labels):
    #     print(tokenizer.decode([id]), label)
    #
    # predicted_labels = [model.config.id2label[id] for id in predicted_label_classes.squeeze().tolist()]
    # print(predicted_labels)
    #
    # for id, label in zip(encoding.input_ids.squeeze().tolist(), predicted_labels):
    #     print(tokenizer.decode([id]), label)
    #
    # for prediction, label in zip(predictions, labels):
    #     for predicted_idx, label_idx in zip(prediction, label):
    #         if label_idx == -100:
    #             continue
    #         all_predictions.append(label_names[predicted_idx])
    #         all_labels.append(label_names[label_idx])
print(all_predictions)
print(all_labels)

X_test = all_predictions
y_test = all_labels

new_x_test = []
new_y_test = []


for a in all_predictions:
    new_x_test.append(label2id[a])
for a in all_labels:
    new_y_test.append(label2id[a])

X_test = new_x_test
y_test = new_y_test

x_predicted = X_test
y_test = y_test
#print(x_predicted)
#print(y_test)
a = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5:0}
b = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5:0}
for i in x_predicted:
    a[int(i)] += 1
for i in y_test:
    b[int(i)] += 1
print(a)
print(b)

print("BERT Accuracy Score -> ", accuracy_score(y_test, x_predicted))

print("BERT Precision Score (AVERAGE_NONE)-> ", precision_score(y_test, x_predicted, average=None))
print("BERT Precision Score (AVERAGE_micro)-> ", precision_score(y_test, x_predicted, average='micro'))
print("BERT Precision Score (AVERAGE_macro)-> ", precision_score(y_test, x_predicted, average='macro'))
print("BERT Precision Score (AVERAGE_weighted)-> ", precision_score(y_test, x_predicted, average='weighted'))

print("BERT Recall Score(AVERAGE_NONE) -> ", recall_score(y_test, x_predicted, average=None))
print("BERT Recall Score(AVERAGE_micro) -> ", recall_score(y_test, x_predicted, average='micro'))
print("BERT Recall Score(AVERAGE_macro) -> ", recall_score(y_test, x_predicted, average='macro'))
print("BERT Recall Score(AVERAGE_weighted) -> ", recall_score(y_test, x_predicted, average='weighted'))

print("BERT F1 Score (AVERAGE_NONE)-> ", f1_score(y_test, x_predicted, average=None))
print("BERT F1 Score (AVERAGE_micro)-> ", f1_score(y_test, x_predicted, average='micro'))
print("BERT F1 Score (AVERAGE_macro)-> ", f1_score(y_test, x_predicted, average='macro'))
print("BERT F1 Score (AVERAGE_weighted)-> ", f1_score(y_test, x_predicted, average='weighted'))

# print("BERT Log_loss -> ", log_loss(y_test, x_predicted))
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sn

matrix = confusion_matrix(y_test, x_predicted)
df_cm = pd.DataFrame(matrix, index=[i for i in label2id],
                     columns=[i for i in label2id])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()



ax = plt.subplot()
sn.heatmap(df_cm, annot=True, fmt='g', ax=ax);  # annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')

ax.set_title('Confusion Matrix - BERT Tagger')
plt.tight_layout()
plt.show()

tokenizer.save_pretrained("./models/tokenizer/")

exit()
