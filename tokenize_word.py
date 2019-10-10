"""
describe: the module is used to extract the text feature into number-form vector
input: the text
output: the tf-idf feature vectors
"""

import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


# part1: download the stopwords and punctuation in English, to tokenize the English-based text

nltk.download("punkt")
nltk.download("stopwords")
stop_words = nltk.corpus.stopwords.words("english")
print(stop_words[:10])

stemmer = nltk.stem.snowball.SnowballStemmer("english")
print(stemmer)


# part2: tokenize the English Text to form the stem
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# part3: use tf-idf method to extract the text features
def get_tfidf_vectorizer():
    # tf-idf algorithm in here
    # define vectorizer parameters
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                       min_df=0.01, stop_words='english',
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))
    # tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) #fit the vectorizer to synopses

    return tfidf_vectorizer


def train_tfidf(text_list):
    # use tf-idf method to train the vector
    tfidf_vectorizer = get_tfidf_vectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform(text_list)  # fit the vectorizer to synopses
    terms = tfidf_vectorizer.get_feature_names()

    print(terms)
    print(tfidf_matrix.shape)

    return tfidf_matrix

