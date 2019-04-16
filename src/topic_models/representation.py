from __future__ import division

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from gensim.topic_coherence import text_analysis
from gensim.corpora import dictionary
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec

from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing as pp
from similarity import calculate_sims, global_measures
import pickle
from collections import defaultdict
import inspect


class Represent(object):

    """

    """

    def __init__(self, dictionary=None, texts=None):
        self.dictionary = dictionary
        self.occurrences = self.dictionary.dfs
        self.build_cooccurrences()
        self.texts = texts
        self.w2vattrs =[]

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

    def build_word2vec(self, size = 100, sg=0, window=10):
        #path = get_tmpfile("word2vec.model")
        w2vmodel = Word2Vec(self.texts, size=size,sg=sg, window=20, min_count=1, workers=4)
        #w2vmodel.wv.save("word2vec.model")
        method = ['cbow','skipgram']
        attrname ='w2v%s%s%s' %(size, method[sg], window)
        self.w2vattrs.append(attrname)
        setattr(self, attrname, w2vmodel.wv)

    def build_many_w2v(self, sizes=None, windows =None):
        for size in sizes:
            for window in windows:
                self.build_word2vec(size=size, sg=1, window=window)
                self.build_word2vec(size=size, sg=0, window=window)

    def cosine_matrix(self, matrix):
        return cosine_similarity(matrix)

    def build_similarity_matrix(self, similarity, function):
        pass

    def n_most_similar(self, word,  n=10, build = False):
        """
        Given a word or vector, what are the n most similar word/vectors
        :param word: word or vector
        :param n: number of results to return
        :return: top n most similar words (as strings)
        """
        sims =defaultdict(lambda: defaultdict(list))
        id = self.dictionary.token2id[word]
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
                        #so far only cosine distance, using sklearn and sparse is fastest
                        vector,matr = self.cooc[id], self.cooc
                        sorted = np.argsort(function(vector, matr))
                        topn = sorted[0,:n]
                        sims['indirect'][name]=[self.dictionary[match] for match in topn]
                    else:
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
                        sims['direct'][name] = astokens[::-1]
        if self.w2vattrs:
            for w2vmodel in self.w2vattrs:
                wv = getattr(self, w2vmodel)
                sims['indirect'][w2vmodel]= [w for w,score in wv.most_similar(word, topn=n)]
        simsreform = {(outerKey, innerKey): values for outerKey, innerDict in sims.iteritems()
                  for innerKey, values in innerDict.iteritems()}

        return simsreform

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
