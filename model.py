import tensorflow as tf
import numpy as np
from data_utils import *


class Model(object):

    def _init_(dim_hidden, dim_embedding, batch_size, sentence_length, vocab_size):
        self._dim_hidden = dim_hidden
        self._dim_embedding = dim_embedding
        self._batch_size = batch_size
        self._sentence_length = sentence_length
        self._vocab_size = vocab_size

    def LSTM():
        embedding = tf.Variable(
            tf.random_uniform([self._vocab_size, self._dim_embedding], -1.0, 1.0))
