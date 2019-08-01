# -*- coding: utf-8 -*-

import pandas as pd
import os
import argparse
import time
import jieba
import pickle
import re
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


MAX_VOCAB_SIZE = 10000


# code form https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
def clean_en_text(dat):

    REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')

    ret = []
    for line in dat:
        # text = text.lower() # lowercase text
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = BAD_SYMBOLS_RE.sub('', line)
        line = line.strip()
        ret.append(line)
    return ret

def clean_zh_text(dat):
    REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')

    ret = []
    for line in dat:
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = line.strip()
        ret.append(line)
    return ret


def _tokenize_chinese_words(text):
    return ' '.join(jieba.cut(text, cut_all=False))


def vectorize_data(x_train, x_val=None):
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features = MAX_VOCAB_SIZE)
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


# onhot encode to category
def ohe2cat(label):
    return np.argmax(label, axis=1)


class Model(object):
    """ Example of valid model """

    def __init__(self, metadata):
        """ Initialization for model
        :param metadata: a dict formed like:
            {"class_num": 10,
             "language": ZH,
             "num_train_instances": 10000,
             "num_test_instances": 1000,
             "time_budget": 300}
        """
        self.done_training = False
        self.metadata = metadata
        self.train_output_path = './'
        self.test_input_path = './'

    def train(self, train_dataset, remaining_time_budget=None):
        """Train this algorithm on the NLP task.

         This method will be called REPEATEDLY during the whole training/predicting
         process. So your `train` method should be able to handle repeated calls and
         hopefully improve your model performance after each call.

         ****************************************************************************
         ****************************************************************************
         IMPORTANT: the loop of calling `train` and `test` will only run if
             self.done_training = False
           (the corresponding code can be found in ingestion.py, search
           'M.done_training')
           Otherwise, the loop will go on until the time budget is used up. Please
           pay attention to set self.done_training = True when you think the model is
           converged or when there is not enough time for next round of training.
         ****************************************************************************
         ****************************************************************************

        :param train_dataset: tuple, (x_train, y_train)
            x_train: list of str, input training sentence.
            y_train: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:

        :return: None
        """
        if self.done_training:
            return
        x_train, y_train = train_dataset

        # tokenize Chinese words
        if self.metadata['language'] == 'ZH':
            x_train = clean_zh_text(x_train)
            x_train = list(map(_tokenize_chinese_words, x_train))
        else:
            x_train = clean_en_text(x_train)

        x_train, tokenizer = vectorize_data(x_train)
        model = LinearSVC(random_state=0, tol=1e-5)
        model.fit(x_train, ohe2cat(y_train))

        with open(self.train_output_path + 'model.pickle', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.train_output_path + 'tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.done_training=True


    def test(self, x_test, remaining_time_budget=None):
        """
        :param x_test: list of str, input test sentence.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                 here `sample_count` is the number of examples in this dataset as test
                 set and `class_num` is the same as the class_num in metadata. The
                 values should be binary or in the interval [0,1].
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
