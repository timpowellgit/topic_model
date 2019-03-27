
from itertools import combinations

def segment_ordered(topics):

    return [list(combinations(t,2)) for t in topics]