from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from gensim.topic_coherence import text_analysis
from gensim.corpora import dictionary
from scipy.sparse import coo_matrix


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

        self.cooc = self.build_cooccurrences()

    def build_cooccurrences(self):
        # try:
        #     cooc_dict = self.dictionary.cooc_dict
        # except AttributeError:
        cooc_dict =self.dictionary.cooc_dict
        rows, cols, data = zip(*[(row, col, cooc_dict[row][col]) for row in cooc_dict for col in cooc_dict[row]])
        coocm = coo_matrix((data, (rows, cols)))
        coocm.resize(len(self.occurrences),len(self.occurrences))

        return coocm.T + coocm
