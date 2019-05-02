from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional, concatenate, \
    CuDNNLSTM, CuDNNGRU, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Input, \
    Flatten, GRU
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import roc_auc_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import os
from symspellpy.symspellpy import SymSpell, Verbosity
from utils import process_tweet, under_sample
import numpy as np
import pandas as pd
import sklearn
import time
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# NLP
import re
import string
import html
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')

# Setting Flags
balance_dataset = True             # If true, it under-samples the training dataset to get same amount of labels
use_pretrained_embeddings = True    # If true, it enables the use of GloVe pre-trained Twitter word-embeddings


#########################################################################################
# 1. LOAD EMBEDDINGS AND BUILD EMBEDDINGS INDEX                                         #
#########################################################################################
embed_size = 100

if use_pretrained_embeddings:
    path = Path("embedding_index.pkl")
    if path.is_file():
        with open("embedding_index.pkl", "rb") as f:
            embedding_index = pickle.load(f)
    else:
        # Download embeddings from https://nlp.stanford.edu/projects/glove/
        #                          https://nlp.stanford.edu/data/glove.twitter.27B.zip
        embedding_path = Path("glove.twitter.27B.200d.txt")

        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        # Construct embedding table (word -> vector)
        print("Building embedding index [word->vector]", end="\n")
        t0 = time.time()
        embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path, encoding="utf8"))

        with open("embedding_index.pkl", "wb") as f:
            pickle.dump(embedding_index, f)

        print(" - Done! ({:0.2f}s)".format(time.time() - t0))

#########################################################################################
# 2. LOAD TWEET DATA AND PRE-PROCESS                                                    #
#########################################################################################
params = dict(remove_USER_URL=True,
              remove_stopwords=True,
              remove_HTMLentities=True,
              remove_punctuation=True,
              appostrophe_handling=True,
              lemmatize=True,
              reduce_lengthenings=True,
              segment_words=True,
              correct_spelling=True
             )

print("Loading training data")
train_balanced = Path('train_balanced.csv')
trial_balanced = Path('trial_balanced.csv')
train_clean = Path('train_clean.csv')
trial_clean = Path('trial_clean.csv')

if balance_dataset and train_balanced.exists() and trial_balanced.exists():
    train_data = train_balanced
    trial_data=trial_balanced
elif train_clean.exists and trial_clean.exists():
    train_data = train_clean
    trial_data=trial_clean
else:
    train_data = Path('start-kit/training-v1/offenseval-training-v1.tsv')
    trial_data = Path('start-kit/trial-data/offenseval-trial.txt')

df_a = pd.read_csv(train_data, sep='\t')
df_a_trial = pd.read_csv(trial_data, sep='\t')#,names=['id','tweet','subtask_a','subtask_b', 'subtask_c', ])
print("Done!")
print("Preprocessing...")

df_a.tweet = df_a['tweet'].apply(lambda x: process_tweet(x, **params, trial=False))
df_a.subtask_a = df_a['subtask_a'].replace({'OFF': 1, 'NOT': 0})

train_tweet = df_a.tweet.values
train_label = df_a.subtask_a.values

class_weights = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(train_label), train_label.reshape(-1))

df_a_trial['tweet'] = df_a_trial['tweet'].apply(lambda x: process_tweet(x, **params, trial=True))
df_a_trial['subtask_a'] = df_a_trial['subtask_a'].replace({'OFF': 1, 'NOT': 0})

trial_tweet = df_a_trial['tweet'].values
trial_label = df_a_trial['subtask_a'].values

print("Done!")

if balance_dataset:
    df_a.to_csv(train_clean)
    df_a_trial.to_csv(trial_clean)
    train_tweet, train_label = under_sample(train_tweet, train_label)
    df_a.to_csv(train_balanced)
    df_a_trial.to_csv(trial_balanced)
else:
    df_a.to_csv(train_clean)
    df_a_trial.to_csv(trial_clean)

# print("EXAMPLES OF PROCESSED TWEETS [train/trial]")
# print("_________________________________________________________________________________________________________")
# for id in range(10, 15):
#     print("Un-processed:  " + df_a['tweet'][id])
#     print("Processed:     " + X[id])
#     print("")
# print("_________________________________________________________________________________________________________")
# for id in range(10, 15):
#     print("Un-processed:  " + df_a_trial['tweet'][id])
#     print("Processed:     " + trial_tweet[id])
#     print("")

#########################################################################################
# 3. BUILD VOCABULARY FROM FULL CORPUS AND PREPARE INPUT                                #
#    Tokenize tweets | Turn into Index sequences | Pad sequences | Word embeddings      #
#########################################################################################
max_seq_len = 50
max_features = 30000

# Tokenize all tweets
tokenizer = Tokenizer(lower=True, filters='', split=' ')
X_all = list(train_tweet) + list(trial_tweet)
tokenizer.fit_on_texts(X_all)
print(f"Num of unique tokens in tokenizer: {len(tokenizer.word_index)}")

# Get sequences for each dataset
sequences = tokenizer.texts_to_sequences(train_tweet)
sequences_trial = tokenizer.texts_to_sequences(trial_tweet)

# Pad sequences
train_tweet = pad_sequences(sequences, maxlen = max_seq_len)
trial_tweet = pad_sequences(sequences_trial, maxlen = max_seq_len)

# Reshape labels
train_label = train_label.reshape(-1,1)
trial_label = trial_label.reshape(-1,1)

if use_pretrained_embeddings:
    # Build Embedding Matrix
    n_words_in_glove = 0
    n_words_not_in_glove = 0
    words_in_glove = []
    words_not_in_glove = []

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, embed_size))
    print(f"Building embedding matrix {embedding_matrix.shape}", end="")
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            n_words_in_glove += 1
            words_in_glove.append(word)
        else:
            n_words_not_in_glove += 1
            words_not_in_glove.append(word)
    print(" - Done!")
    print("This vocabulary has {} unique tokens of which {} are in the embeddings and {} are not".format(len(word_index), n_words_in_glove,
                                                                                                         n_words_not_in_glove))
    print(f"Words not in Glove: {len(words_not_in_glove)}")

# Show sentence length frequency plot
sentence_lengths = [len(tokens) for tokens in sequences]
print("Mean sentence length: {:0.1f} words".format(np.mean(sentence_lengths)))
print("MAX  sentence length: {} words".format(np.max(sentence_lengths)))

fig, ax = plt.subplots(nrows=1, ncols=1)

fig.set_size_inches([20, 8])

ax.set_title('Sentence lengths', fontsize=30)
ax.set_xlabel('Tweet length', fontsize=30)
ax.set_ylabel('Number of Tweets', fontsize=30)
ax.hist(sentence_lengths, bins=list(range(70)))
ax.tick_params(labelsize=20)
fig.savefig("sentence_lenghts.pdf", bbox_inches="tight")

# SET HYPERPARAMETERS
LR               = 0.004
LR_DECAY         = 0
EPOCHS           = 20
BATCH_SIZE       = 32
EMBEDDING_DIM    = embed_size
DROPOUT          = 0.4         # Connection drop ratio for CNN to LSTM dropout
LSTM_DROPOUT     = 0.0         # Connection drop ratio for gate-specific dropout
BIDIRECTIONAL    = True
RECURRENT_UNITS  = 100
train_embeddings = not use_pretrained_embeddings

#########################################################################################
# 3. BUILD AND TRAIN THE MODEL                                                          #
#########################################################################################
class ROC_F1(Callback):
    def __init__(self, validation_data=(), training_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.X_train, self.y_train = training_data
        self.f1s_train = []
        self.f1s_val = []
        self.aucs_train = []
        self.aucs_val = []

    def on_epoch_end(self, epoch, logs={}):
        lr = self.model.optimizer.lr
        if self.model.optimizer.initial_decay > 0:
            lr = lr * (1. / (1. + self.model.optimizer.decay * K.cast(self.model.optimizer.iterations, K.dtype(self.model.optimizer.decay))))
        if epoch % self.interval == 0:
            y_pred_train = np.round(self.model.predict(X_train, verbose=0))
            y_pred_val = np.round(self.model.predict(self.X_val, verbose=0))

            auc_train = roc_auc_score(y_train, y_pred_train)
            auc_val = roc_auc_score(self.y_val, y_pred_val)
            f1_train = f1_score(y_train, y_pred_train, average='macro')
            f1_val = f1_score(self.y_val, y_pred_val, average='macro')

            self.aucs_train.append(auc_train)
            self.aucs_val.append(auc_val)
            self.f1s_val.append(f1_val)
            self.f1s_train.append(f1_train)

            print("     - LR: {:0.5f} train_auc: {:.4f} - train_F1: {:.4f} - val_auc: {:.4f} - val_F1: {:.4f}".format(K.eval(lr), auc_train, f1_train,
                                                                                                                      auc_val, f1_val))
        print("\n\n")

def build_Bi_GRU_LSTM_CN_model(lr=LR, lr_decay=LR_DECAY, recurrent_units=RECURRENT_UNITS, dropout=DROPOUT):
    # Model architecture
    inputs = Input(shape=(max_seq_len,), name="Input")

    emb = Embedding(nb_words + 1, embed_size, trainable=train_embeddings, name="WordEmbeddings")(inputs)
    emb = SpatialDropout1D(dropout)(emb)

    gru_out = Bidirectional(CuDNNGRU(RECURRENT_UNITS, return_sequences=True), name="Bi_GRU")(emb)
    gru_out = Conv1D(32, 4, activation='relu', padding='valid', kernel_initializer='he_uniform')(gru_out)

    lstm_out = Bidirectional(CuDNNLSTM(RECURRENT_UNITS, return_sequences=True), name="Bi_LSTM")(emb)
    lstm_out = Conv1D(32, 4, activation='relu', padding='valid', kernel_initializer='he_uniform')(lstm_out)

    avg_pool1 = GlobalAveragePooling1D(name="GlobalAVGPooling_GRU")(gru_out)
    max_pool1 = GlobalMaxPooling1D(name="GlobalMAXPooling_GRU")(gru_out)

    avg_pool2 = GlobalAveragePooling1D(name="GlobalAVGPooling_LSTM")(lstm_out)
    max_pool2 = GlobalMaxPooling1D(name="GlobalMAXPooling_LSTM")(lstm_out)

    x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])

    outputs = Dense(1, activation='sigmoid', name="Output")(x)

    model = Model(inputs, outputs)

    return model, 1


def build_LSTM():
    model = Sequential()
    model.add(Embedding(nb_words + 1, embed_size, input_length=max_seq_len, trainable=train_embeddings, name="Embeddings"))
    model.add(SpatialDropout1D(DROPOUT))
    model.add(Bidirectional(LSTM(RECURRENT_UNITS)))
    model.add(Dense(1, activation='sigmoid'))
    return model, 0


def build_CNN_LSTM():
    EMBEDDING_DIM = embed_size
    model = Sequential()
    model.add(Embedding(nb_words + 1, EMBEDDING_DIM, input_length=max_seq_len, trainable=train_embeddings, name="Embeddings"))
    model.add(SpatialDropout1D(DROPOUT))
    model.add(Conv1D(64, 4, activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Bidirectional(LSTM(RECURRENT_UNITS, dropout=LSTM_DROPOUT, recurrent_dropout=LSTM_DROPOUT)))
    model.add(Dense(1, activation='sigmoid'))
    return model, 0


def build_LSTM_CNN():
    EMBEDDING_DIM = embed_size
    model = Sequential()
    model.add(Embedding(nb_words + 1, EMBEDDING_DIM, input_length=max_seq_len, trainable=train_embeddings, name="Embeddings"))
    #     model.add(SpatialDropout1D(DROPOUT))
    model.add(Dropout(DROPOUT))
    model.add(Bidirectional(CuDNNLSTM(RECURRENT_UNITS, return_sequences=True)))
    #     model.add(Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True, dropout=LSTM_DROPOUT, recurrent_dropout=LSTM_DROPOUT)))
    model.add(Conv1D(64, kernel_size=2, activation='relu', padding='valid', kernel_initializer='he_uniform'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model, 0
import inspect
print(inspect.signature(build_Bi_GRU_LSTM_CN_model))

def doit(model, embed_idx, type=0):

    # ----------------------


    # BUILD MODEL
    # - Select which architecture to use (simple LSTM works well)

    # OPTIMIZER | COMPILE | EMBEDDINGS
    optim = optimizers.Adam(lr=LR, decay=LR_DECAY)
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
    if use_pretrained_embeddings:
        model.layers[embed_idx].set_weights([embedding_matrix])
    model.summary()

    X_train, X_val, y_train, y_val = train_test_split(train_tweet, train_label, test_size=0.2, stratify=train_label)

    class_weights = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(y_train), y_train.reshape(-1))
    weights_dict = dict()
    for i, weight in enumerate(class_weights):
        weights_dict[i] = weight
    print("Class weights (to address dataset imbalance):")

    # FIT THE MODEL ------------------------------------------------------------------------------------------------
    auc_f1 = ROC_F1(validation_data=(X_val, y_val), training_data=(X_train, y_train), interval=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto', restore_best_weights=True)
    filepath = "weights-improvement-{epoch:02d}-{val_acc:.5f}-{val_loss:.5f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', verbose=1, mode='min')

    train_history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS,
                            verbose=1, class_weight=class_weights, callbacks=[earlystop, checkpoint, auc_f1])
    model.save(f"taskA_model_{type}.h5")
    # ---------------------------------------------------------------------------------------------------------------


    #########################################################################################
    # 4. EVALUATE MODEL (LOSS PROFILE, F1-SCORES, CONFUSION MATRIX, AUC,                    #
    #########################################################################################
    height = 3.5
    width = height * 4
    n_epochs = 40
    # n_epochs = len(train_history.history['loss'])

    # Plot Loss
    plt.figure(figsize=(width,height))
    plt.plot(train_history.history['loss'], label="Train Loss")
    plt.plot(train_history.history['val_loss'], label="Validation Loss")
    plt.xlim([0,n_epochs-1]); plt.xticks(list(range(n_epochs)));   plt.grid(True);   plt.legend()
    plt.title("Loss (Binary Cross-entropy)", fontsize=15)
    plt.show()

    # Plot accuracy
    plt.figure(figsize=(width,height))
    plt.plot(train_history.history['acc'], label="Train Accuracy")
    plt.plot(train_history.history['val_acc'], label="Validation Accuracy")
    plt.xlim([0,n_epochs-1]); plt.xticks(list(range(n_epochs)));   plt.grid(True);   plt.legend()
    plt.title("Accuracy", fontsize=15)
    plt.show()

    # Plot F1
    plt.figure(figsize=(width, height))
    plt.plot(auc_f1.f1s_train, label="Train F1")
    plt.plot(auc_f1.f1s_val, label="Validation F1")
    plt.xlim([0, n_epochs - 1]);
    plt.xticks(list(range(n_epochs)));
    plt.grid(True);
    plt.legend()
    plt.title("F1-score", fontsize=15)
    plt.show()

    # Plot ROC AUC
    plt.figure(figsize=(width, height))
    plt.plot(auc_f1.aucs_train, label="Train ROC AUC")
    plt.plot(auc_f1.aucs_val, label="Validation ROC AUC")
    plt.xlim([0, n_epochs - 1]);
    plt.xticks(list(range(n_epochs)));
    plt.grid(True);
    plt.legend()
    plt.legend()
    plt.title("ROC AUC", fontsize=15)
    plt.show()


    def plot_confusion_matrix(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    X_eval = trial_tweet
    y_eval = trial_label

    y_pred = model.predict(X_eval)
    y_pred = np.round(y_pred)
    print("Validation Accuracy: {:0.2f}%".format(np.sum(y_eval == y_pred) / y_eval.shape[0] * 100))

    # cm = confusion_matrix(y_eval, y_pred)
    # fig = plt.figure(figsize=(10, 10))
    # plot = plot_confusion_matrix(cm, classes=['NOT-OFFENSIVE', 'OFFENSIVE'], normalize=True, title='Confusion matrix')
    # plt.show()
    # print(cm)

    print(classification_report(y_eval, y_pred))



modelfuncs = [build_Bi_GRU_LSTM_CN_model, build_CNN_LSTM, build_LSTM_CNN,  build_LSTM, ]
for func,i in zip(modelfuncs, range(len(modelfuncs))):
    model, embed_idx = func()
    sig = inspect.signature(func)
    print(sig)
    doit(model, embed_idx, i)

# model, embed_idx =

# model, embed_idx = build_CNN_LSTM()
# model, embed_idx = build_LSTM_CNN()
# model, embed_idx = build_Bi_GRU_LSTM_CN_model(LR, LR_DECAY, RECURRENT_UNITS, DROPOUT)