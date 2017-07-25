# -*- coding: utf-8 -*-
from keras.layers import Input, Activation
from keras.layers.core import Flatten, Dense
from keras.layers.merge import Dot
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.losses import mean_squared_error
from keras.metrics import binary_accuracy
import keras.backend as K
import utils_extension as utils
from cilin import CilinSimilarity
import tensorflow as tf
import random

# text file containing sentences where words are separated
data_file = '../ganyan_sentence_clean.txt'

def batch_generator(cpl, lbl):

    # trim the tail
    garbage = len(labels) % batch_size

    pvt = cpl[:, 0][:-garbage]
    ctx = cpl[:, 1][:-garbage]
    lbl_ctx = lbl[:, 0][:-garbage]
    lbl_cilin = lbl[:, 1][:-garbage]

    assert pvt.shape == ctx.shape == lbl_ctx.shape == lbl_cilin.shape

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
            yield ([pvt[begin: end], ctx[begin: end]], lbl[begin: end])
            
            


# load data
# - sentences: list of (list of word-id)
# - index2word: list of string
sentences, index2word = utils.load_sentences(data_file, 100)
# params
nb_epoch = 3
# learn `batch_size words` at a time
batch_size = 10
vec_dim = 50
# half of window
window_size = 5
vocab_size = len(index2word)

# create input
couples, labels = utils.skip_grams_with_cilin(index2word, sentences, window_size, vocab_size)
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

merged = Dot(axes=-1)([embedded_pvt, embedded_ctx])

flatten = Flatten()(merged)
dense = Dense(2)(flatten)
predictions = Activation('sigmoid')(dense)

# y_true - {(0,1]: distance, 0:unknown}
def mean_squared_error_cilin(y_true, y_pred, nonzero_count):
    index = K.not_equal(y_true, 0)
    index = K.cast(index, y_true.dtype)
    y_pred_ = tf.multiply(y_pred, index)
    loss_cilin = tf.reduce_sum(K.square(y_pred_ - y_true))
    nonzero_count = K.cast(nonzero_count, loss_cilin.dtype)
    loss_cilin = tf.divide(loss_cilin, nonzero_count)
    return loss_cilin

def zero_loss():
    return 0.0

def loss_with_cilin(y_true, y_pred):
  # loss for skip-gram model
  y_true_ctx = y_true[:, 0]
  y_pred_ctx = y_pred[:, 0]
  loss_sg = mean_squared_error(y_true_ctx, y_pred_ctx)
  # loss for cilin
  y_true_cilin = y_true[:, 1]
  y_pred_cilin = y_pred[:, 1]
  # obtain indexes of elements with value equal to 0
  nonzero_count = tf.count_nonzero(y_true_cilin)
  loss_cilin = tf.cond(tf.greater(nonzero_count, 0),
            lambda: mean_squared_error_cilin(y_true_cilin, y_pred_cilin, nonzero_count),
            lambda: zero_loss())
  loss = tf.add(loss_sg, loss_cilin)
  return loss

def absolute_accuracy_cilin(y_true, y_pred, nonzero_count):
    index = K.not_equal(y_true, 0)
    index = K.cast(index, y_true.dtype)
    y_pred_ = tf.multiply(y_pred, index)
    accuracy_cilin = tf.reduce_sum(K.abs(y_true - y_pred_))
    nonzero_count = K.cast(nonzero_count, accuracy_cilin.dtype)
    accuracy_cilin = tf.divide(accuracy_cilin, nonzero_count)
    return 1 - accuracy_cilin

def metric_with_cilin(y_true, y_pred):
  # loss for skip-gram model
  y_true_ctx = y_true[:, 0]
  y_pred_ctx = y_pred[:, 0]
  accuracy_sg = binary_accuracy(y_true_ctx, y_pred_ctx)
  # loss for cilin
  y_true_cilin = y_true[:, 1]
  y_pred_cilin = y_pred[:, 1]
  # obtain indexes of elements with value equal to 0
  nonzero_count = tf.count_nonzero(y_true_cilin)
  accuracy_cilin = tf.cond(tf.greater(nonzero_count, 0),
            lambda: absolute_accuracy_cilin(y_true_cilin, y_pred_cilin, nonzero_count),
            lambda: zero_loss())
  weights = tf.constant([0.5, 0.5]) 
  accuracy = tf.reduce_sum(tf.multiply(weights, tf.stack([accuracy_sg, accuracy_cilin])))
  return accuracy
       

# build and train the model
model = Model(inputs=[input_pvt, input_ctx], outputs=predictions)
#model.summary()
model.compile(optimizer='rmsprop', loss=loss_with_cilin, metrics=[metric_with_cilin])
model.fit_generator(generator=batch_generator(couples, labels),
                    steps_per_epoch=samples_per_epoch,
                    epochs=nb_epoch, verbose=1)



# save weights
#utils.save_weights(model, index2word, vec_dim)
