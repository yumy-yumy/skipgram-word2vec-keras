# -*- coding: utf-8 -*-
from keras.layers import Input, Activation
from keras.layers.core import Flatten, Dense
from keras.layers.merge import Dot
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.losses import mean_squared_error
import keras.backend as K
import utils_extension as utils
import numpy as np
import tensorflow as tf


def batch_generator(cpl, lbl):
    import random

    # trim the tail
    garbage = len(labels) % batch_size

    pvt = cpl[:, 0][:-garbage]
    ctx = cpl[:, 1][:-garbage]
    #lbl = lbl[:-garbage]
    lbl_ctx = lbl[:, 0][:-garbage]
    lbl_label = lbl[:, 1][:-garbage]

    #assert pvt.shape == ctx.shape == lbl.shape
    assert pvt.shape == ctx.shape == lbl_ctx.shape == lbl_label.shape

    # epoch loop
    while 1:
        # shuffle data at beginning of every epoch (takes few minutes)
        seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(pvt)
        random.seed(seed)
        random.shuffle(ctx)
        random.seed(seed)
        random.shuffle(lbl)

        for i in range(nb_batch):
            begin, end = batch_size*i, batch_size*(i+1)
            # feed i th batch
            #yield ([pvt[begin: end], ctx[begin: end]], lbl[begin: end])
            yield ([pvt[begin: end], ctx[begin: end]], [lbl_ctx[begin: end], lbl_label[begin: end]])



# load data
# - sentences: list of (list of word-id)
# - index2word: list of string
#sentences, index2word = utils.load_sentences_brown()
sentences, index2word = utils.load_sentences('../ganyan_jieba_clean.txt', 5)

# params
nb_epoch = 3
# learn `batch_size words` at a time
batch_size = 10
vec_dim = 50
# half of window
window_size = 5
vocab_size = len(index2word)

# create input
couples, labels = utils.skip_grams_with_label(index2word, sentences, window_size, vocab_size)
print 'shape of couples: ', couples.shape
print 'shape of labels: ', labels.shape


# metrics
nb_batch = len(labels) // batch_size
samples_per_epoch = batch_size * nb_batch


# graph definition (pvt: center of window, ctx: context)
input_pvt = Input(batch_shape=(batch_size, 1), dtype='int32')
input_ctx = Input(batch_shape=(batch_size, 1), dtype='int32')

embedded_pvt = Embedding(input_dim=vocab_size,
                         output_dim=vec_dim)(input_pvt)

embedded_ctx = Embedding(input_dim=vocab_size,
                         output_dim=vec_dim)(input_ctx)

merged = Dot(axes=2)([embedded_pvt, embedded_ctx])

flatten = Flatten()(merged)

prediction = Activation('sigmoid', name='prediction')(flatten)
prediction_label = Activation('sigmoid', name='prediction_label')(flatten)

# loss for a true label 
# y_true = {1: synonym, 0: not synonym, -1:unknown}
def mean_squared_error_label(y_true, y_pred, nonzero_count):
    index = K.not_equal(y_true, -1)
    index = K.cast(index, y_true.dtype)
    y_true_ = tf.multiply(y_true, index)
    y_pred_ = tf.multiply(y_pred, index)
    loss = tf.reduce_sum(K.square(y_pred_ - y_true_))
    nonzero_count = K.cast(nonzero_count, loss.dtype)
    loss = tf.divide(loss, nonzero_count)
    return loss

def zero_loss():
    return 0.0

def loss_with_label(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    # obtain indexes of elements with value equal to -1
    index = K.not_equal(y_true, -1)
    nonzero_count = tf.count_nonzero(index)
    loss = tf.cond(tf.greater(nonzero_count, 0),
            lambda: mean_squared_error_label(y_true, y_pred, nonzero_count),
            lambda: zero_loss())

    return loss

# binary accuracy for a true label indicating whether two words are synonyms
def binary_accuracy_label(y_true, y_pred, nonzero_count):
    accuracy = K.equal(y_true, K.round(y_pred))
    accuracy = K.cast(accuracy, tf.float32)
    accuracy = tf.reduce_sum(accuracy)
    nonzero_count = K.cast(nonzero_count, accuracy.dtype)
    accuracy = tf.divide(accuracy, nonzero_count)
    return accuracy

def metric_with_label(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    # obtain indexes of elements with value equal to -1
    index = K.not_equal(y_true, -1)
    nonzero_count = tf.count_nonzero(index)
    accuracy = tf.cond(tf.greater(nonzero_count, 0),
            lambda: binary_accuracy_label(y_true, y_pred, nonzero_count),
            lambda: zero_loss())

    return accuracy
        

# build and train the model
model = Model(inputs=[input_pvt, input_ctx], outputs=[prediction, prediction_label])
#model.summary()
model.compile(optimizer='rmsprop', loss=['mse', loss_with_label], 
                metrics=['accuracy'])
model.fit_generator(generator=batch_generator(couples, labels),
                    steps_per_epoch=samples_per_epoch,
                    epochs=nb_epoch, verbose=1)


# save weights
#utils.save_weights(model, index2word, vec_dim)

