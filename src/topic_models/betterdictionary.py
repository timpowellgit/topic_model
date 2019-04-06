

from gensim.corpora.dictionary import Dictionary

from collections import Mapping, defaultdict
from itertools import combinations

from six import PY3, iteritems, iterkeys, itervalues, string_types
from six.moves import zip, range

class BetterDictionary(Dictionary):

    def __init__(self, documents=None, prune_at=2000000, cooccurrences =True):
        #super(BetterDictionary,self).__init__(documents=documents,prune_at=prune_at)
        self.token2id = {}
        self.id2token = {}
        self.dfs = {}

        self.num_docs = 0
        self.num_pos = 0
        self.num_nnz = 0

        self.count_cooccurrences = cooccurrences
        self.cooc_dict = defaultdict(lambda: defaultdict(int))

        if documents is not None:
            self.add_documents(documents, prune_at=prune_at)


    def doc2bow(self, document, allow_update=False, return_missing=False):
        """Convert `document` into the bag-of-words (BoW) format = list of `(token_id, token_count)` tuples.

        Parameters
        ----------
        document : list of str
            Input document.
        allow_update : bool, optional
            Update self, by adding new tokens from `document` and updating internal corpus statistics.
        return_missing : bool, optional
            Return missing tokens (tokens present in `document` but not in self) with frequencies?

        Return
        ------
        list of (int, int)
            BoW representation of `document`.
        list of (int, int), dict of (str, int)
            If `return_missing` is True, return BoW representation of `document` + dictionary with missing
            tokens and their frequencies.
        """
        if isinstance(document, string_types):
            raise TypeError("doc2bow expects an array of unicode tokens on input, not a single string")

        # Construct (word, frequency) mapping.
        counter = defaultdict(int)
        for w in document:
            counter[w if isinstance(w, unicode) else unicode(w, 'utf-8')] += 1

        token2id = self.token2id
        if allow_update or return_missing:
            missing = sorted(x for x in iteritems(counter) if x[0] not in token2id)
            if allow_update:
                for w, _ in missing:
                    # new id = number of ids made so far;
                    # NOTE this assumes there are no gaps in the id sequence!
                    token2id[w] = len(token2id)
        result = {token2id[w]: freq for w, freq in iteritems(counter) if w in token2id}

        if allow_update:
            self.num_docs += 1
            self.num_pos += sum(itervalues(counter))
            self.num_nnz += len(result)
            # increase document count for each unique token that appeared in the document
            # and increase cooccurrence count for each ordered combo
            dfs = self.dfs
            cooc_dict= self.cooc_dict
            if self.count_cooccurrences:
                #abc -> ab ac bc... cba -> ab ac bc with sorted
                tokenids = sorted(iterkeys(result))
                for i in range(len(tokenids) - 1):
                    for j in range(i + 1, len(tokenids)):
                        first = tokenids[i]
                        second = tokenids[j]
                        cooc_dict[first][second]+=1
                    dfs[tokenids[i]]= dfs.get(tokenids[i], 0) + 1
                #for loop ignores last one
                dfs[tokenids[-1]]= dfs.get(tokenids[-1], 0) + 1
            else:
                for tokenid in iterkeys(result):
                    dfs[tokenid] = dfs.get(tokenid, 0) + 1

        # return tokenids, in ascending id order
        result = sorted(iteritems(result))
        if return_missing:
            return result, dict(missing)
        else:
            return result