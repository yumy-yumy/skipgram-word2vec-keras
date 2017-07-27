# -*- coding: utf-8 -*-
import numpy as np
from cilin import CilinSimilarity
from ioFile import dataFromFile, dataToFile, save_object, load_object

# output file
filename = 'vectors.txt'

# an object containing the cilin dictionary
cilin_file = 'object.pkl'

def load_sentences_brown(nb_sentences=None):
    """
    :param nb_sentences: Use if all brown sentences are too many
    :return: index2word (list of string)
    """
    from nltk.corpus import brown
    from gensim.models import word2vec

    print 'building vocab ...'

    if nb_sentences is None:
        sents = brown.sents()
    else:
        sents = brown.sents()[:nb_sentences]
        
    sent_list = [[s.lower() for s in sent] for sent in sents]
    sents = sent_list

    print len(sents)
    
    # I use gensim model only for building vocab
    model = word2vec.Word2Vec()
    model.build_vocab(sents)
    vocab = model.wv.vocab
    print len(vocab)

    # ids: list of (list of word-id)
    ids = [[vocab[w].index for w in sent
            if w in vocab and vocab[w].sample_int > model.random.rand() * 2**32]
           for sent in sents]
    
    
    return ids, model.wv.index2word
    
def load_sentences(fname, nb_sentences=None):
    """
    :param nb_sentences: Use if all brown sentences are too many
    :return: index2word (list of string)
    """
    from gensim.models import word2vec

    print 'building vocab ...'

    data_iterator = dataFromFile(fname)
    i = 0
    sents = []
    for line in data_iterator:  
    #    line = line.encode('utf-8')
        line = line.rstrip('\n')
        sents.append(line.split(' '))
        i += 1
        if nb_sentences is not None:
            if nb_sentences == i :
                break
    print "load", i, "sentences"
    # I use gensim model only for building vocab
    model = word2vec.Word2Vec()
    model.build_vocab(sents)
    vocab = model.wv.vocab
    print "vocabulary size is", len(vocab)

    # ids: list of (list of word-id)
    ids = [[vocab[w].index for w in sent
            if w in vocab and vocab[w].sample_int > model.random.rand() * 2**32]
           for sent in sents]
    
    
#    save_object('sentences.pkl', ids)
#    save_object('index2word.pkl', model.wv.index2word)
    return ids, model.wv.index2word


def skip_grams(sentences, window, vocab_size, nb_negative_samples=5.):
    """
    calc `keras.preprocessing.sequence.skipgrams` for each sentence
    and concatenate those.

    :param sentences: list of (list of word-id)
    :return: concatenated skip-grams
    """
    import keras.preprocessing.sequence as seq
    import numpy as np

    print 'building skip-grams ...'

    def sg(sentence):
        return seq.skipgrams(sentence, vocab_size,
                             window_size=np.random.randint(window - 1) + 1,
                             negative_samples=nb_negative_samples)

    couples = []
    labels = []

    # concat all skipgrams
    for cpl, lbl in map(sg, sentences):
        couples.extend(cpl)
        labels.extend(lbl)
        print len(cpl)
        break

    return np.asarray(couples), np.asarray(labels)

def skip_grams_with_cilin(index2word, sentences, window, vocab_size, nb_negative_samples=5.):
    import keras.preprocessing.sequence as seq
    import numpy as np
    
    print 'building skip-grams and labels...'

    def sg(sentence):
        return seq.skipgrams(sentence, vocab_size,
                            window_size=np.random.randint(window - 1) + 1,
                            negative_samples=nb_negative_samples)

    couples = []
    labels = []

    # concat all skipgrams
    for cpl, lbl in map(sg, sentences):
        couples.extend(cpl)
        labels.extend(lbl)

    
    cs = load_object('object.pkl')
    
    cilin_dist = []
    for word, context_word in couples: 
        sim = cs.similarity(index2word[word], index2word[context_word])
        cilin_dist.append(sim)
        if len(cilin_dist) % 10000 == 0:
            print len(cilin_dist)
     
    return np.asarray(couples), np.asarray([labels, cilin_dist]).T
    
def skip_grams_with_cilin_for_big_data(index2word, sentences, window, vocab_size, nb_negative_samples=1.):
    file_segment_size = 10000
    i = 0
    j = 0
    data_size = 0
    while i < len(sentences):
        if i + file_segment_size < len(sentences):
            couples, labels = utils.skip_grams_with_cilin(index2word, sentences[i:i+file_segment_size], window_size, vocab_size)
        else:
            couples, labels = utils.skip_grams_with_cilin(index2word, sentences[i:], window_size, vocab_size)
        '''
        fname = 'data/couples/couples_' + str(j) + '.pkl'
        save_object(fname, couples)
        fname = 'data/labels/labels_' + str(j) + '.pkl'
        save_object(fname, labels)
        '''
        data_size += len(labels)
        print "processed", i, "sentences"
        i += file_segment_size
        j += 1
    
    return data_size
     
def skip_grams_with_label(index2word, sentences, window, vocab_size, nb_negative_samples=5.):
    import keras.preprocessing.sequence as seq
    
    print 'building skip-grams and labels...'

    def sg(sentence):
        return seq.skipgrams(sentence, vocab_size,
                            window_size=np.random.randint(window - 1) + 1,
                            negative_samples=nb_negative_samples)

    couples = []
    labels = []

    # concat all skipgrams
    for cpl, lbl in map(sg, sentences):
        couples.extend(cpl)
        labels.extend(lbl)

    
    true_label = load_object('labels.pkl')

    return np.asarray(couples), np.asarray([labels, true_label]).T 

def save_weights(model, index2word, vec_dim):
    """
    :param model: keras model
    :param index2word: list of string
    :param vec_dim: dim of embedding vector
    :return:
    """
    vec = model.get_weights()[0]
    f = open(filename, 'w')
    # first row in this file is vector information
    f.write(" ".join([str(len(index2word)), str(vec_dim)]))
    f.write("\n")
    for i, word in enumerate(index2word):
        f.write(word.encode('utf-8'))
        f.write(" ")
        f.write(" ".join(map(str, list(vec[i, :]))))
        f.write("\n")
    f.close()


def most_similar(positive=[], negative=[]):
    """
    :param positive: list of string
    :param negative: list of string
    :return:
    """
    from gensim import models

    n_top = 5
    vec = models.KeyedVectors.load_word2vec_format('vectors.txt', binary=False)
    for v in vec.most_similar_cosmul(positive=positive, negative=negative, topn=n_top):
        print v[0].decode('utf-8'), v[1]

def most_similar_cos(positive=[], negative=[], sg_model=None, vocab=None, index2word=None):
    weight = sg_model.get_weights()[0]
    word_vector = np.zeros(weight[0, :].shape)
    for w in positive:
      word_vector += weight[vocab[w].index, :]
    
    print word_vector[:2]
    vec_norm = np.zeros(word_vector.shape)
    d = np.sum(word_vector**2,)**(0.5)
    vec_norm = (word_vector.T / d).T
    print vec_norm[:2]
    dist = np.dot(weight, vec_norm)
    for w in positive:
      index = vocab[w].index
      dist[index] = -np.Inf
    
    word_index_list = np.argsort(-dist)[:n_top]
    for index in word_index_list:
      print index2word[index], dist[index]
  