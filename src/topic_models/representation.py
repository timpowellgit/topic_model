from __future__ import division

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from gensim.topic_coherence import text_analysis
from gensim.corpora import dictionary
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing as pp
"""
GENSIM VS SKLEARN INHERIT?
Gensim due to easier preprocessing and dicts will be consistent
, faster too?
sliding window option
    also has tf idf, normalizing, etc
    
    
NEITHER 
just use dicoitnary, ge toccurences and coocc (text analysis expects topic ids, we wanna run it before)

Is init override needed??

can i easily get pmi matrix, pmi +2 -2 context vectors??  pmi sentence vectors?

w tf-idf
    sublinear scaling is default true? occurences in doc = 1+ log(24) (instead of say 24)
    
    smoothing default true? In sklearn, smooting is add one document with one of each term
        may interfere w various pmi smoothing techniques
"""

class Represent(object):

    """

    """

    def __init__(self, dictionary=None):
        self.dictionary = dictionary
        self.occurrences = self.dictionary.dfs
        self.build_cooccurrences()

    def build_cooccurrences(self):
        # try:
        #     cooc_dict = self.dictionary.cooc_dict
        # except AttributeError:
        cooc_dict =self.dictionary.cooc_dict
        rows, cols, data = zip(*[(row, col, cooc_dict[row][col]) for row in cooc_dict for col in cooc_dict[row]])
        coocm = coo_matrix((data, (rows, cols))).tocsr()
        coocm.resize(len(self.occurrences),len(self.occurrences))

        self.cooc = coocm.T + coocm

    def build_pmi_matrix(self):
        pass

    def reduce_matrix(self):
        pass

    def build_word2vec(self):
        pass

    def cosine_matrix(self, matrix):
        return cosine_similarity(matrix)

    def get_sparsity(self, matrix):
        A = matrix.todense()
        return 1.0 - np.count_nonzero(A)/A.size