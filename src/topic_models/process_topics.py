from __future__ import print_function
from gensim.models import coherencemodel
from collections import namedtuple
import pandas as pd
import io
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (segmentation, probability_estimation,
                                    direct_confirmation_measure, indirect_confirmation_measure,
                                    aggregation)
from gensim.parsing.preprocessing import *
from gensim.corpora import Dictionary, csvcorpus
from similarity import calculate_sims
from segmentation import segment_with_weights
from gensim.models import CoherenceModel, nmf, LdaMulticore
from farotate import FARotate
from sklearn.decomposition import FactorAnalysis
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from IPython.display import display

from ipywidgets import interact, interactive, IntSlider, Layout, interact_manual, fixed
import ipywidgets as widgets
import qgrid
import time
import os

from collections import defaultdict
import inspect


_make_pipeline = namedtuple('Coherence_Measure', 'seg, prob, conf, aggr')
NEW_COHERENCE_MEASURES = {
    'all': _make_pipeline(
        segment_with_weights,
        probability_estimation.p_boolean_document,
        None,
        aggregation.arithmetic_mean
    )
}
coherencemodel.COHERENCE_MEASURES.update(NEW_COHERENCE_MEASURES)

NEW_SLIDING_WINDOW_SIZES = {
    'all':None
}
coherencemodel.SLIDING_WINDOW_SIZES.update(NEW_SLIDING_WINDOW_SIZES)

NEW_BOOLEAN_DOCUMENT_BASED = {'all'}
coherencemodel.BOOLEAN_DOCUMENT_BASED.update(NEW_BOOLEAN_DOCUMENT_BASED)



class NewCoherence(coherencemodel.CoherenceModel):

    def __init__(self, topics=None, corpus=None,dictionary=None,coherence=None, cooccurrence=None, tf_vectorizer = None):
        super(NewCoherence,self).__init__(topics=topics, corpus=corpus,dictionary=dictionary,coherence=coherence)
        self.cooccurrence = cooccurrence
        self.tf_vectorizer = tf_vectorizer

    def get_all_coherences_per_topic(self,segmented_topics=None):
        measure = self.measure
        weights = None
        if segmented_topics is None:
            if measure.seg == segment_with_weights:
                self.segmented_topics, self.weights = measure.seg(self.topics)
            else:
                self.segmented_topics = measure.seg(self.topics)
        if self._accumulator is None:
            self.estimate_probabilities(segmented_topics)


        return self.get_all_measures(measures_list= 'all', weights= weights)

    def _ensure_elements_are_ids(self, topic):
        return np.array([self.dictionary.token2id[token] for token in topic])


    def get_all_measures(self,measures_list=None, weights=None):

        topic_coherences = []
        num_docs = float(self._accumulator.num_docs)
        for topic_index,segments_i in enumerate(self.segmented_topics):
            segments_sims = defaultdict(list)
            for w_prime, w_star in segments_i:
                w_prime_count = self._accumulator[w_prime]
                w_star_count = self._accumulator[w_star]
                co_occur_count = self._accumulator[w_prime, w_star]
                co_profiles = self.get_cooccurrence_profiles(w_prime, w_star)
                sims = calculate_sims(w_prime_count,w_star_count,co_occur_count,co_profiles, num_docs,measures_list)
                for measure, score in sims:
                    segments_sims[measure].append(score)
            for dict_measure, score_list in segments_sims.items():

                avg = np.average(score_list)#,weights=weights[topic_index] if weights else None)
                wavg = np.average(score_list,weights=self.weights[topic_index] if self.weights else None)

                segments_sims[dict_measure] = wavg
            topic_coherences.append(segments_sims)
        return topic_coherences


    def get_cooccurrence_profiles(self, w,w2):
        #return 'dummy','dummy'
        tokens = self.dictionary[w], self.dictionary[w2]
        indices = self.tf_vectorizer.vocabulary_[tokens[0]], self.tf_vectorizer.vocabulary_[tokens[1]]
        coprofile, coprofile_star = self.cooccurrence[indices[0]], self.cooccurrence[indices[1]]

        return coprofile[0],coprofile_star[0]

def coherence_scores(coherence, topics, corpus, dictionary, cooccur, tf_vectorizer):
    cm = NewCoherence(topics=topics, corpus=corpus, dictionary=dictionary, 
        coherence=coherence, cooccurrence= cooccur, tf_vectorizer= tf_vectorizer)
    #model_score = cm.get_coherence()
    topic_coherences = cm.get_all_coherences_per_topic()
    return topic_coherences

def get_gensim_topics(model, n_topics):
    topics = []
    for topic_id, topic in model.show_topics(num_topics=n_topics, formatted=False):
        topic = [word for word, _ in topic]
        topics.append(topic)
    return topics

def get_sklearn_topics(model, n_topics, feature_names):
    topics= []
    for topic_idx, topic in enumerate(model.components_):
        topics.append([feature_names[i] for i in topic.argsort()])
    return topics


def run_all(data, model_type, n_topics=10, coherence='all'):
    topics = None
    texts, dictionary, corpus,tf_vectorizer, tf, cooccur = data
    if (model_type == 'LDA'):

        lda = LdaMulticore(corpus=corpus,
                           num_topics=n_topics,
                           id2word=dictionary,passes=5)
        topics = get_gensim_topics(lda, n_topics)

    elif (model_type == 'FA'):
        famodel = FARotate(n_components=n_topics, rotation='varimax')
        famodel.fit(tf.toarray())
        topics = get_sklearn_topics(famodel, n_topics, tf_vectorizer.get_feature_names())
    elif (model_type == 'NMF'):
        nmfmodel = nmf.Nmf(
            corpus=corpus,
            num_topics=n_topics,
            id2word=dictionary,
            chunksize=2000,
            passes=5,
            random_state=42,
        )
        topics = get_gensim_topics(nmfmodel, n_topics)


    coherences = coherence_scores(coherence, topics, corpus, dictionary, cooccur, tf_vectorizer)
    topics= [{"Topic":" ".join(topic)} for topic in topics]
    topicsdf =pd.DataFrame(data=topics)
    coherencesdf = pd.DataFrame(data=coherences)
    both = pd.concat([topicsdf, coherencesdf.round(4)], axis =1 )
    pd.set_option('display.max_colwidth', 200)
    #display(both)
    col_options = {
        'width': 70,
    }
    col_defs = {
        'Topic': {
            'width': 560,
        }
    }

    show = qgrid.show_grid(both,
                    column_options=col_options,
                    column_definitions=col_defs,
                    grid_options={'forceFitColumns': False, 'maxVisibleRows': 100})
    display(show)
    return topics,coherences, both


def coherence_widget(data):
    style = {'description_width': 'initial'}
    m = interactive(run_all, {'manual': True, 'manual_name': 'Run'},
            data=fixed(data),
            model_type=widgets.RadioButtons(options=['NMF','FA', 'LDA'],
                                            description="Choose Model", style=style,
                                            layout=Layout(width='250px')),
            n_topics=widgets.IntSlider(min=1, max=100, step=1, description='Number of topics',
                                       style=style, value = 10, continuous_update=False),
            # coherence=widgets.SelectMultiple(options=['u_mass', 'c_v', 'c_uci', 'c_nmpi','all'],
            #                                  description='Coherence Measure',
            #                                  disabled=False)

            )

    return m



def show_topic_words(topics):
    for i, topic in enumerate(topics):
        print('\n', i, end=" ")
        for word in topic:
            print(word, end=' ')



# if __name__ == '__main__':
    # print('running')

    # plots = []
    # with open('../data/movieplotsawk') as f:
    #     plots =f.readlines()


    # print('read texts')

    # plots = [i.split() for i in plots]
    # dictionarymovie = Dictionary(plots)
    # corpusmovie = [dictionarymovie.doc2bow(text) for text in plots]

    # tmfile =open('../data/topicsMovie.txt')
    # topicsmovie = [i.rstrip('\n').split() for i in tmfile.readlines()]
    # print(dictionarymovie.token2id['strip'])
    # moviecoherence = coherence_scores(coherence='all', corpus= corpusmovie, dictionary =dictionarymovie, topics= topicsmovie)
    # print(moviecoherence)




#     elections = io.open('Election2008Paragraphes.txt', encoding="ISO-8859-1")
#     e = elections.readlines()
#     CUSTOM_FILTERS = [lambda x: x.lower(), strip_punctuation, strip_multiple_whitespaces, strip_numeric,
#                       remove_stopwords, strip_short]
#
#     texts = [preprocess_string(line, filters=CUSTOM_FILTERS) for line in e[:100]]
#     dictionary = Dictionary(texts)
#     corpus = [dictionary.doc2bow(text) for text in texts]
#     tf_vectorizer = CountVectorizer()
#     tftexts  =[' '.join(text) for text in texts]
#     tf = tf_vectorizer.fit_transform(tftexts)
#     tf= tf.toarray()
#     start =time.time()
#     fa = FARotate(n_components=10,rotation='varimax')
#     fa.fit(tf)
#     print('farotate total time',time.time()-start)
#     start2 = time.time()
#     fa= FactorAnalysis(n_components=10)
#     fa.fit(tf)
#     print('fa total time',time.time()-start2)
#
#     start3 = time.time()
#     fa= FactorAnalysisCopy(n_components=10,rotation='varimax')
#     fa.fit(tf)
#     print('fa total time',time.time()-start3)
#     #
#     # print(fa)
#     # print(fa.components_)
#     # data = [texts, dictionary, corpus]
#     # print('test')
#     # run_all(data,'NMF')