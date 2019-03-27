from __future__ import print_function
from gensim.models import coherencemodel
from collections import namedtuple
import pandas as pd
import pyLDAvis
import io
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (segmentation, probability_estimation,
                                    direct_confirmation_measure, indirect_confirmation_measure,
                                    aggregation)
from gensim.parsing.preprocessing import *
from gensim.corpora import Dictionary, csvcorpus

from similarity import all_measures
from segmentation import segment_ordered
from gensim.models import CoherenceModel, nmf, LdaMulticore
from farotate import FARotate
from farotatecopy import FactorAnalysisCopy
from sklearn.decomposition import FactorAnalysis
from sklearn.feature_extraction.text import CountVectorizer

from IPython.display import display

from ipywidgets import interact, interactive, IntSlider, Layout, interact_manual, fixed
import ipywidgets as widgets
import qgrid
import time

_make_pipeline = namedtuple('Coherence_Measure', 'seg, prob, conf, aggr')
NEW_COHERENCE_MEASURES = {
    'all': _make_pipeline(
        segment_ordered,
        probability_estimation.p_boolean_document,
        all_measures,
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

    def get_all_coherences_per_topic(self,segmented_topics=None):
        measure = self.measure
        if segmented_topics is None:
            segmented_topics = measure.seg(self.topics)
        if self._accumulator is None:
            self.estimate_probabilities(segmented_topics)

        return measure.conf(segmented_topics, self._accumulator, measures_list= 'all')

def get_gensim_topics(model, n_topics):
    topics = []
    for topic_id, topic in model.show_topics(num_topics=n_topics, formatted=False):
        topic = [word for word, _ in topic]
        print(topic)
        topics.append(topic)
    return topics

def get_sklearn_topics(model, n_topics, feature_names):
    topics= []
    for topic_idx, topic in enumerate(model.components_):
        topics.append([feature_names[i] for i in topic.argsort()])
    return topics


def run_all(data, model_type, n_topics=10, coherence='all'):
    topics = None
    texts, dictionary, corpus = data
    if (model_type == 'LDA'):

        lda = LdaMulticore(corpus=corpus,
                           num_topics=n_topics,
                           id2word=dictionary,passes=5)
        topics = get_gensim_topics(lda, n_topics)

    elif (model_type == 'FA'):
        tf_vectorizer = CountVectorizer()
        tftexts = [' '.join(text) for text in texts]
        tf = tf_vectorizer.fit_transform(tftexts)
        tf_feature_names = tf_vectorizer.get_feature_names()
        tf = tf.toarray()
        famodel = FARotate(n_components=n_topics, rotation='varimax')
        famodel.fit(tf)
        topics = get_sklearn_topics(famodel, n_topics, tf_feature_names)
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

    def coherence_scores(coherence, topics):
        cm = NewCoherence(topics=topics, corpus=corpus, dictionary=dictionary, coherence=coherence)
        #model_score = cm.get_coherence()
        topic_coherences = cm.get_all_coherences_per_topic()
        return topic_coherences

    coherences = coherence_scores(coherence, topics)
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