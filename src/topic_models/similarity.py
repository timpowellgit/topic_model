from __future__ import division

import numpy as np
from collections import defaultdict
import inspect
import logging, sys



EPSILON = 1e-12


def sim_jaccard(w,w2,co):
    return co/((w+w2)-co)


def sim_dice(w,w2,co):
    return 2*co/(2*(w+w2-co))

def sim_chisquare(w,w2,co,num_docs):
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
    if w2==0:
        return 0.0
    lc = np.math.log(((co / num_docs) + EPSILON) / (w2 / num_docs))
    return lc

def sim_inclusion(w,w2,co):
    return co/min(w,w2)

def sim_association(w,w2,co):
    return (co*co)/ (w*w2)

def sim_npmi(w,w2,co,num_docs):
    if co==0:
        return 0.0
    else:
        pmi = np.math.log((co/num_docs) / (w/num_docs) * (w2/num_docs))
        npmi = pmi/np.math.log(co/num_docs)
    return npmi

def sim_pmi(w,w2,co,num_docs):
    V = np.math.log(((co/num_docs) + EPSILON) / (w/num_docs) * (w2/num_docs))
    return V


global_measures = [(name, measure_function) for name, measure_function in locals().items() if name.startswith('sim_') ]


def calculate_sims(w,w2,co,num_docs,measures):
    if measures == 'all':
        sim_scores = []
        for measure_name, measure_func in global_measures:
            if 'num_docs' in inspect.getargspec(measure_func).args:
                score = measure_func(w,w2,co,num_docs)
            else:
                score = measure_func(w,w2,co)
            #print('score calculated on ', measure_name, w,w2,co,num_docs,"score ", score  )
            sim_scores.append((measure_name[4:],score))

        return sim_scores


def all_measures(segmented_topics,accumulator, measures_list = None):

    topic_coherences = []
    num_docs = float(accumulator.num_docs)
    for s_i in segmented_topics:
        segment_sims = defaultdict(list)
        for w_prime, w_star in s_i:
            w_prime_count = accumulator[w_prime]
            w_star_count = accumulator[w_star]
            co_occur_count = accumulator[w_prime, w_star]
            sims = calculate_sims(w_prime_count,w_star_count,co_occur_count, num_docs,measures_list)
            for measure, score in sims:
                segment_sims[measure].append(score)
        for dict_measure, score_list in segment_sims.items():
            avg = np.mean(score_list)
            segment_sims[dict_measure] = avg
        topic_coherences.append(segment_sims)
    return topic_coherences

