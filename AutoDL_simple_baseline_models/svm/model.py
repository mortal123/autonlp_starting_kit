import pandas as pd
import os
import argparse
import time
import jieba
import pickle
import tensorflow as tf
import numpy as np
import sys, getopt
from subprocess import check_output
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from tensorflow.python.keras.preprocessing import text
from tensorflow.python.keras.preprocessing import sequence


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


def _tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def _tokenize_chinese_words(text):
    return ' '.join(jieba.cut(text, cut_all=False))


def vectorize_data(x_train, x_val=None):
    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    if x_val:
        full_text = x_train + x_val
    else:
        full_text = x_train
    vectorizer.fit(full_text)
    train_vectorized = vectorizer.transform(x_train)
    if x_val:
        val_vectorized = vectorizer.transform(x_val)
        return train_vectorized, val_vectorized, vectorizer
    return train_vectorized, vectorizer


def OHE_to(label):
    return np.argmax(label, axis=1)


class Model(object):
    """Trivial example of valid model. Returns all-zero predictions."""

    def __init__(self, metadata, train_output_path="./", test_input_path="./"):
        """

        :param metadata: a dict which contains these k-v pair: language, num_train_instances, num_test_instances, xxx.
        :param train_output_path: a str path contains training model's output files, including model.pickle and tokenizer.pickle.
        :param test_input_path: a str path contains test model's input files, including model.pickle and tokenizer.pickle.
        """
        self.done_training = False
        self.metadata = metadata
        self.train_output_path = train_output_path
        self.test_input_path = test_input_path

    def train(self, train_dataset, remaining_time_budget=None):
        """

        :param x_train: list of str, input training sentence.
        :param y_train: list of lists of int, sparse input training labels.
        :param remaining_time_budget:
        :return:
        """
        if self.done_training:
            return

        x_train, y_train = train_dataset

        # tokenize Chinese words
        if self.metadata['language'] == 'ZH':
            x_train = list(map(_tokenize_chinese_words, x_train))

        x_train, tokenizer = vectorize_data(x_train)
        model = LinearSVC(random_state=0, tol=1e-5)
        print(str(type(x_train)) + " " + str(y_train.shape))
        model.fit(x_train, OHE_to(y_train))

        with open(self.train_output_path + 'model.pickle', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.train_output_path + 'tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.done_training = True

    def test(self, x_test, remaining_time_budget=None):
        """
        :param x_test: list of str, input test sentence.
        :param remaining_time_budget:
        :return: list of lists of int, sparse output model prediction labels.
        """
        with open(self.test_input_path + 'model.pickle', 'rb') as handle:
            model = pickle.load(handle, encoding='iso-8859-1')
        with open(self.test_input_path + 'tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle, encoding='iso-8859-1')

        train_num, test_num = self.metadata['train_num'], self.metadata['test_num']
        class_num = self.metadata['class_num']

        # tokenizing Chinese words
        if self.metadata['language'] == 'ZH':
            x_test = list(map(_tokenize_chinese_words, x_test))

        x_test = tokenizer.transform(x_test)
        result = model.predict(x_test)

        # category class list to sparse class list of lists
        y_test = np.zeros([test_num, class_num])
        for idx, y in enumerate(result):
            y_test[idx][y] = 1
        return y_test


if __name__ == '__main__':
    path = '/Users/a/Documents/autonlp_datasets/thucnews_public/'
    df = pd.read_csv(path + 'cnews.dev.txt', sep='\t',
                     names=['labels', 'contents'])
    x_train = list(df['contents'].values)
    le = LabelEncoder()
    le.fit(df['labels'])
    y_train = le.transform(df['labels'])
    model = Model_SVM('a', '/Users/a/Documents/baseline_autonlp_test/', '/Users/a/Documents/baseline_autonlp_test/')
    model.train(x_train, y_train)
    res = model.test(x_train)
