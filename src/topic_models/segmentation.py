
from itertools import combinations

def segment_ordered(topics):

    return [list(combinations(t,2)) for t in topics]


def segment_with_weights(topics):
    combos_all= []
    weights_all = []
    for topic in topics:
        combos = []
        weights= []
        for i in range(len(topic)-1):
            for j in range(i+1, len(topic)):
                weight = len(topic)-i + len(topic)-j
                combos.append((topic[i],topic[j]))
                weights.append(weight)
        combos_all.append(combos)
        weights_all.append(weights)
    return combos_all, weights_all