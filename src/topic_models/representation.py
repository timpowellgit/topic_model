from __future__ import division

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from gensim.topic_coherence import text_analysis
from gensim.corpora import dictionary
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing as pp
from similarity import calculate_sims, global_measures
import pickle
from collections import defaultdict
import inspect

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
        cooc_dict =self.dictionary.cooc_dict
        rows, cols, data = zip(*[(row, col, cooc_dict[row][col]) for row in cooc_dict for col in cooc_dict[row]])
        coocm = coo_matrix((data, (rows, cols))).tocsr()
        coocm.resize(len(self.occurrences),len(self.occurrences))
        self.cooc = coocm.T + coocm

        #set diagonal to occurences
        self.occs = [self.occurrences[i] for i in xrange(len(self.occurrences))]
        self.cooc.setdiag(self.occs)
        self.coocdense = self.cooc.todense()

    def build_pmi_matrix(self):
        pass

    def reduce_matrix(self):
        pass

    def build_word2vec(self):
        pass

    def cosine_matrix(self, matrix):
        return cosine_similarity(matrix)

    def build_similarity_matrix(self, similarity, function):
        pass
        # if 'profiles' in inspect.getargspec(function).args:
        #
        #     #score = measure_func(co_profiles, num_docs)
        #     pass
        # elif 'num_docs' in inspect.getargspec(function).args:
        #
        #     score = function(w, w2, co, num_docs)
        #
        # else:
        #     score = measure_func(w, w2, co)

    def n_most_similar(self, word,  n=10, build = False):
        """
        Given a word or vector, what are the n most similar word/vectors
        :param word: word or vector
        :param n: number of results to return
        :return: top n most similar words (as strings)
        """
        id = self.dictionary.token2id[word]
        print global_measures
        #try to get instantiated/saved similarity matrix per measure
        for name, function in global_measures:
            sim_matrix = "%s_matrix" %name
            try:
                similarity_matrix = getattr(self, sim_matrix)
            except AttributeError:
                sim_pickled = "%s.pickle" %sim_matrix
                try:
                    similarity_matrix= self.load_similarity_matrix(sim_pickled)
                except IOError:
                    if build:
                        self.build_similarity_matrix(name, function)
                        similarity_matrix = getattr(self, sim_matrix)
                    elif name.endswith('ind'):
                        print name
                        sorted = np.argsort(function(self.coocdense[id], self.coocdense[:]))
                        topn = sorted[0,:n]
                        print [self.dictionary[match] for match in topn]
                    else:
                        print name
                        #use functional programming to apply sim function to arrays
                        num_args = function.func_code.co_argcount
                        ufunc = np.frompyfunc(function, num_args, 1)
                        occs = self.occs
                        query_occs = [occs[id]]* len(self.dictionary)
                        co_w_id = self.coocdense[id]
                        num_docs = self.dictionary.num_docs
                        args = [query_occs,occs,co_w_id,num_docs]
                        sorted = np.argsort(ufunc(*args[:num_args]))
                        topn = sorted[0,-n:]
                        astokens = [self.dictionary[match] for match in topn.flat]
                        #reverse it
                        print astokens[::-1]

                        print

    def save_similarity_matrix(self, filename, matrix):
        with open(filename, 'wb') as handle:
            pickle.dump(matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_similarity_matrix(self, filename):
        with open(filename, 'rb') as handle:
            matrix = pickle.load(handle)

        return matrix



    def get_sparsity(self, matrix):
        A = matrix.todense()
        return 1.0 - np.count_nonzero(A)/A.size