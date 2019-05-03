import html
import html
import re
import string
from random import shuffle
from pathlib import Path
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer, word_tokenize
from sklearn.metrics import f1_score, roc_auc_score
from symspellpy import SymSpell, Verbosity
from tensorflow.keras.callbacks import Callback
from nltk.corpus import *
from nltk.corpus.reader.wordnet import *

nltk.download('wordnet')
nltk.download('stopwords')
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7,
                 count_threshold=1, compact_level=0)



# sym_spell.create_dictionary(Path('english_words_479k.txt'))

sym_spell.load_dictionary(Path('./frequency_dictionary_en_82_765.txt'), 0,1)

def process_tweet(data,
                  remove_USER_URL=True,
                  remove_punctuation=True,
                  remove_stopwords=True,
                  remove_HTMLentities=True,
                  remove_hashtags=True,
                  appostrophe_handling=True,
                  lemmatize=True,
                  trial=False,
                  reduce_lengthenings=True,
                  segment_words=True,
                  correct_spelling=True,
                  create_dict=True
                  ):
    # print(type(tweet))
    if type(data) is not str():
        id = data[0]
        tweet = data[1]
    else:
        tweet = data

    """
    This function receives tweets and returns clean word-list
    """
    ### Handle USERS and URLS ################################################
    if remove_USER_URL:
        if trial:
            tweet = re.sub(r'@\w+ ?', '', tweet)
            tweet = re.sub(r'http\S+', '', tweet)
        else:
            tweet = re.sub(r"@USER", "<>", tweet)
            tweet = re.sub(r"URL", "", tweet)
    else:
        if trial:
            tweet = re.sub(r'@\w+ ?', '<usertoken> ', tweet)
            tweet = re.sub(r'http\S+', '<urltoken> ', tweet)
        else:
            tweet = re.sub(r"@USER", "<usertoken>", tweet)
            tweet = re.sub(r"URL", "<urltoken>", tweet)
    ### Remove HTML Entiti es #################################################
    if remove_HTMLentities:
        tweet = html.unescape(tweet)

    ### REMOVE HASHTAGS? #####################################################
    if remove_hashtags:
        tweet = re.sub(r'#\w+ ?', '', tweet)

    ### Convert to lower case: Hi->hi, MAGA -> maga ##########################
    tweet = tweet.lower()

    ### Cleaning: non-ASCII filtering, some appostrophes, separation #########
    tweet = re.sub(r"’", r"'", tweet)
    tweet = re.sub(r"[^A-Za-z0-9'^,!.\/+-=@]", " ", tweet)
    tweet = re.sub(r"what's", "what is ", tweet)
    tweet = re.sub(r"\'s", " ", tweet)
    tweet = re.sub(r"\'ve", " have ", tweet)
    tweet = re.sub(r"n't", " not ", tweet)
    #     tweet = re.sub(r"i'm", "i am ", tweet)
    tweet = re.sub(r"\'re", " are ", tweet)
    tweet = re.sub(r"\'d", " would ", tweet)
    tweet = re.sub(r"\'ll", " will ", tweet)
    tweet = re.sub(r",", " ", tweet)
    tweet = re.sub(r"\.", " ", tweet)
    tweet = re.sub(r"!", " ! ", tweet)
    tweet = re.sub(r"\/", " ", tweet)
    tweet = re.sub(r"\^", " ^ ", tweet)
    tweet = re.sub(r"\+", " + ", tweet)
    tweet = re.sub(r"\-", " - ", tweet)
    tweet = re.sub(r"\=", " = ", tweet)
    tweet = re.sub(r"(\d+)(k)", r"\g<1>000", tweet)
    tweet = re.sub(r":", " : ", tweet)
    tweet = re.sub(r" e g ", " eg ", tweet)
    tweet = re.sub(r" b g ", " bg ", tweet)
    tweet = re.sub(r" u s ", " american ", tweet)
    tweet = re.sub(r"\0s", "0", tweet)
    tweet = re.sub(r" 9 11 ", "911", tweet)
    tweet = re.sub(r"e - mail", "email", tweet)
    tweet = re.sub(r"j k", "jk", tweet)
    tweet = re.sub(r"\s{2,}", " ", tweet)

    ### Remove Punctuation ###################################################
    if remove_punctuation:
        translator = str.maketrans('', '', ''.join(list(set(string.punctuation) - set("'"))))
        tweet = tweet.translate(translator)

    # Tokenize sentence for further word-level processing
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    words = tokenizer.tokenize(tweet)

    ### Apostrophe handling:    you're   -> you are  ########################
    APPO = {"aren't": "are not", "can't": "cannot", "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not",
            "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
            "i'd": "I would", "i'll": "I will", "i'm": "I am", "isn't": "is not", "it's": "it is", "it'll": "it will",
            "i've": "I have", "let's": "let us", "mightn't": "might not", "mustn't": "must not", "shan't": "shall not", "she'd": "she would",
            "she'll": "she will", "she's": "she is", "shouldn't": "should not", "that's": "that is", "there's": "there is", "they'd": "they would",
            "they'll": "they will", "they're": "they are", "they've": "they have", "we'd": "we would", "we're": "we are", "weren't": "were not",
            "we've": "we have", "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have", "where's": "where is",
            "who'd": "who would", "who'll": "who will", "who're": "who are", "who's": "who is", "who've": "who have", "won't": "will not",
            "wouldn't": "would not", "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have", "'re": " are",
            "wasn't": "was not", "we'll": " will"}
    if appostrophe_handling:
        words = [APPO[word] if word in APPO else word for word in words]

    tweet = ' '.join(words)
    words = tokenizer.tokenize(tweet)

    ### Lemmatisation:          drinking -> drink ###########################
    if lemmatize:
        wordnet_corpora = nltk.data.find('corpora/wordnet')
        omwReader = LazyCorpusLoader('omw', nltk.corpus.reader.CorpusReader, r'.*/wn-data-.*\.tab', encoding='utf8')
        wnr = WordNetCorpusReader(wordnet_corpora, omwReader)
        wnr.ensure_loaded()
        lemmed = []
        for word in words:
            w = wnr.morphy(word)
            if w:
                lemmed.append(w)
            else:
                lemmed.append(word)
        words = lemmed

    ### Reduce lengthening:    aaaaaaaaah -> aah, bleeeeerh -> bleerh #################
    if reduce_lengthenings:
        pattern = re.compile(r"(.)\1{2,}")
        words = [pattern.sub(r"\1\1", w) for w in words]



    ### Segment words:    thecatonthemat -> the cat on the mat ####################
    if segment_words:
        words = [sym_spell.word_segmentation(word).corrected_string for word in words]

    ### Correct spelling: birberals -> liberals ######################
    if correct_spelling:
        def correct_spelling_for_word(word):
            suggestions = sym_spell.lookup(word, Verbosity.TOP, 2)

            if len(suggestions) > 0:
                return suggestions[0].term
            return word

        words = [correct_spelling_for_word(word) for word in words]
    ### Remove stop words:      is, that, the, ... ##########################
    #### IMPORTANT TO DO THIS LAST SINCE WE MAY MISS SOME WORDS AFTER THEY ARE CLEANED!!#####
    if remove_stopwords:
        eng_stopwords = stopwords.words("english")
        words = set([w for w in words if not w in eng_stopwords])

    clean_tweet = " ".join(words)
    clean_tweet = re.sub("  ", " ", clean_tweet)
    clean_tweet = clean_tweet.lower()

    if type(data) is not str():
        print(f"Finished tweet {id}: {clean_tweet}")
        return [id, clean_tweet]
    print(f'cleaning trial data {trial}:{clean_tweet}')
    if len(clean_tweet) < 1:
        print('empty tweet, returning None!')
        return None

    return clean_tweet


def under_sample(X, y):
    idx_0 = np.where(y == 0)[0].tolist()
    idx_1 = np.where(y == 1)[0].tolist()

    N = np.min([len(idx_0), len(idx_1)])
    idx = idx_0[:N] + idx_1[:N]
    shuffle(idx)

    X = X[idx].reshape(-1)
    y = y[idx].reshape(-1, 1)

    return X, y
