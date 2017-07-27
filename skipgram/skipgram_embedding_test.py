# -*- coding: utf-8 -*-
from gensim import models
from os import path

import utils
import cilin
from cilin import CilinSimilarity
from ioFile import dataFromFile


# evaluate using gensim
test_word_list = [u'乙型', u'肝炎', u'肝癌']
for word in test_word_list:
	print word
	utils.most_similar(positive=[word], negative=[])


