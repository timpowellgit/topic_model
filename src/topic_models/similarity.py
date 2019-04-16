from __future__ import division

import numpy as np
from collections import defaultdict
import inspect
import logging, sys
from sklearn.metrics import pairwise

from scipy.spatial.distance import cdist


EPSILON = 0.000001


def sim_gmean(w,w2,co):
    gmean = co/np.math.sqrt(w*w2)
    return gmean

def sim_zscore(w,w2,co, num_docs):
    a = co
    b = w - a
    c = w2 - a
    d = num_docs - (a + b + c)
    expectation =((a+b) *(a+c))/ num_docs

    if co > expectation:
        a -= 0.5
    elif co < expectation:
        a += 0.5

    zscore = (a - expectation)/np.math.sqrt(expectation)
    return zscore

def sim_jaccard(w,w2,co):
    a = co
    b = w - a
    c = w2 - a
    return a/(a+b+c)


def sim_dice(w,w2,co):
    a = co
    b = w - a
    c = w2 - a
    return 2*a/(2*((a+b)+(a+c)))

def sim_correlation(w,w2,co,num_docs):
    a = co
    b = w -a
    c = w2 -a
    d = num_docs - (a+b+c)

    a2 = ((a+b) *(a+c))/ num_docs
    b2 = ((a+b) *(b+d))/ num_docs
    c2 = ((c+d) *(a+c))/ num_docs
    d2 = ((c+d) *(b+d))/ num_docs
    v = ((a-a2)**2/a2) + ((b-b2)**2/b2) + ((c-c2)**2/c2) + ((d-d2)**2/d2)
    v = v/num_docs

    if v==0:
        return 0
    else:
        return np.math.sqrt(v)


def sim_log_cond(w,w2,co,num_docs):
    #For topic coherence, term list ordered by frequency.
    # Evaluating this measure on its own outside coherence maybe not informative

    if w2==0:
        return 0.0
    lc = np.math.log(((co / num_docs) + EPSILON) / (w2 / num_docs))
    return lc

def sim_inclusion(w,w2,co):
    return co/min(w,w2)

def sim_association(w,w2,co):
    return (co*co)/ (w*w2)

def sim_npmi(w,w2,co,num_docs):
    if co ==0:
        return -1.0
    else:
        pmi = np.math.log((co/num_docs) / ((w/num_docs) * (w2/num_docs)))
        npmi = pmi/-(np.math.log(co/num_docs))
        return npmi

def sim_pmi(w,w2,co,num_docs):
    pmi = np.math.log(((co+1/num_docs) )/(((w+1)/num_docs) * ((w2+1)/num_docs)))

    return pmi

# def sim_pmi2(w,w2,co,num_docs, epsilon = EPSILON):
#     # if not co:
#     #     return 0.0
#     # else:
#     #     pmi = np.math.log((co)/(((w)/num_docs) * ((w2)/num_docs)))

#     # return pmi


def sim_joint_prob(w,w2,co,num_docs):
    return co/num_docs * 100


def sim_cosine_ind(vector, compare):
    cosine =pairwise.cosine_distances(vector, compare)
    return cosine


global_measures = [(name, measure_function) for name, measure_function in locals().items() if name.startswith('sim_') ]


def calculate_sims(w,w2,co,vectors,num_docs,measures):
    if measures == 'all':
        a = co
        b = w - a
        c = w2 - a
        sim_scores = []
        for measure_name, measure_func in global_measures:
            args = inspect.getargspec(measure_func).args
            if 'vector' in args:
                score= measure_func(vectors[0], vectors[1])
            elif 'num_docs' in args:
                score = measure_func(w,w2,co,num_docs)
            else:
                score = measure_func(w,w2,co)
            #print('score calculated on ', measure_name, w,w2,co,num_docs,"score ", score  )
            sim_scores.append((measure_name[4:],score))

        return sim_scores


# def all_measures(segmented_topics,accumulator,cooccurence_matrix, measures_list=None, weights=None):

#     topic_coherences = []
#     num_docs = float(accumulator.num_docs)
#     for topic_index,segments_i in enumerate(segmented_topics):
#         segments_sims = defaultdict(list)
#         for w_prime, w_star in segments_i:
#             w_prime_count = accumulator[w_prime]
#             w_star_count = accumulator[w_star]
#             co_occur_count = accumulator[w_prime, w_star]
#             sims = calculate_sims(w_prime_count,w_star_count,co_occur_count,cooccurence_matrix, num_docs,measures_list)
#             for measure, score in sims:
#                 segments_sims[measure].append(score)
#         for dict_measure, score_list in segments_sims.items():

#             avg = np.average(score_list)#,weights=weights[topic_index] if weights else None)
#             wavg = np.average(score_list,weights=weights[topic_index] if weights else None)

#             segments_sims[dict_measure] = wavg
#         topic_coherences.append(segments_sims)
#     return topic_coherences

