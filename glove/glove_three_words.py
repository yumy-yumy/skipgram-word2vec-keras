# -*- coding: utf-8 -*-
from keras.layers import Input, Activation
from keras.layers.core import Flatten, Lambda
from keras.layers.merge import Dot, Add
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.losses import mean_squared_error
from keras.metrics import binary_accuracy
import keras.backend as K
import tensorflow as tf
import logging
from itertools import izip, combinations
from collections import Counter
from scipy import sparse
import numpy as np
from ioFile import dataFromFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_vocab(corpus):
    """
    Build a vocabulary with word frequencies for an entire corpus.
    Returns a dictionary `w -> (i, f)`, mapping word strings to pairs of
    word ID and word corpus frequency.
    """

    logger.info("Building vocab from corpus")

    vocab = Counter()
    sentences = []
    i = 0
    for line in corpus:
        tokens = line.strip().split()
        vocab.update(tokens)
        sentences.append(tokens)
        i += 1
        if i == 1000:
            break

    logger.info("Done building vocab from corpus.")

    return {word: (i, freq) for i, (word, freq) in enumerate(vocab.iteritems())}, sentences

def build_cooccur(vocab, corpus, window_size=10):
    """
    Build a word co-occurrence list for the given corpus.
    This function is a tuple generator, where each element (representing
    a cooccurrence pair) is of the form
        (i_main, i_context, cooccurrence)
    where `i_main` is the ID of the main word in the cooccurrence and
    `i_context` is the ID of the context word, and `cooccurrence` is the
    `X_{ij}` cooccurrence value as described in Pennington et al.
    (2014).
    If `min_count` is not `None`, cooccurrence pairs where either word
    occurs in the corpus fewer than `min_count` times are ignored.
    """

    vocab_size = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.iteritems())

    # Collect cooccurrences internally as a sparse matrix for passable
    # indexing speed; we'll convert into a list later
    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                      dtype=np.float64)

    for i, line in enumerate(corpus):
        if i % 1000 == 0:
            logger.info("Building cooccurrence matrix: on line %i", i)

#        tokens = line.strip().split()
        tokens = line
        token_ids = [vocab[word][0] for word in tokens]

        for center_i, center_id in enumerate(token_ids):
            # Collect all word IDs in left window of center word
            context_ids = token_ids[max(0, center_i - window_size) : center_i]
            contexts_len = len(context_ids)

            for left_i, left_id in enumerate(context_ids):
                # Distance from center word
                distance = contexts_len - left_i

                # Weight by inverse of distance between words
                increment = 1.0 / float(distance)

                # Build co-occurrence matrix symmetrically (pretend we
                # are calculating right contexts as well)
                cooccurrences[center_id, left_id] += increment
                cooccurrences[left_id, center_id] += increment

    return cooccurrences

def online_generator(cooccurrences, min_count=None):
    # Now yield our tuple sequence (dig into the LiL-matrix internals to
    # quickly iterate through all nonzero cells)
    for k, (row, data) in enumerate(izip(cooccurrences.rows,
                                                   cooccurrences.data)):
        if min_count is not None and vocab[id2word[k]][1] < min_count:
            continue

        for i, j in combinations(row, 2):
            if min_count is not None and vocab[id2word[i]][1] < min_count:
                continue
            if min_count is not None and vocab[id2word[j]][1] < min_count:
                continue          
            yield ([np.array([i]), np.array([j]), np.array([k])], np.array([data[row.index(i)]/data[row.index(j)]]))


# load data
data_iterator = dataFromFile('../ganyan_sentence_clean.txt')
vocab, sentences = build_vocab(data_iterator)

# params
nb_epoch = 3
vec_dim = 50
window_size = 5
vocab_size = len(vocab)
samples_per_epoch = len(sentences)

# create input
coocurrences = build_cooccur(vocab, sentences, window_size)


# graph definition (pvt: center of window, ctx: context)
input_pvt_i = Input(shape=(1,), dtype='int32')
input_pvt_j = Input(shape=(1,), dtype='int32')
input_ctx = Input(shape=(1,), dtype='int32')

embedded_pvt = Embedding(input_dim=vocab_size,
                         output_dim=vec_dim)

embedded_ctx = Embedding(input_dim=vocab_size,
                         output_dim=vec_dim)(input_ctx)

# word i and word j share the same embedding layer
embedded_pvt_i = embedded_pvt(input_pvt_i)
embedded_pvt_j = embedded_pvt(input_pvt_j)

lambda_pvt = Lambda(lambda x: -x)(embedded_pvt_j)

merged_pvt = Add()([embedded_pvt_i, lambda_pvt])

merged = Dot(axes=-1)([merged_pvt, embedded_ctx])

flatten = Flatten()(merged)

prediction = Activation('relu')(flatten)
       

# build and train the model
model = Model(inputs=[input_pvt_i, input_pvt_j, input_ctx], outputs=prediction)
#model.summary()
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
model.fit_generator(generator=online_generator(coocurrences),
                    steps_per_epoch=samples_per_epoch,
                    epochs=nb_epoch, verbose=1)


# save weights
#utils.save_weights(model, index2word, vec_dim)
