from collections import defaultdict
import numpy as np
import math

def _cider_precook(s: str, n: int = 4):
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts

def _cider_cook_refs(refs, n: int = 4):
    # refs: List[str]
    return [_cider_precook(ref, n) for ref in refs]

def _cider_cook_test(test: str, n: int = 4):
    return _cider_precook(test, n)

class CiderScorer(object):
    """CIDEr scorer (no global name collisions)."""

    def __init__(self, refs, test=None, n: int = 4, sigma: float = 6.0,
                 doc_frequency=None, ref_len=None):
        # refs: Dict[id, List[str]]
        # test: Dict[id, List[str]] (each value is list with 1 hypothesis)
        self.n = int(n)
        self.sigma = float(sigma)
        self.crefs = []
        self.ctest = []
        self.doc_frequency = defaultdict(float)
        self.ref_len = None

        # Build cooked references/tests (aligned by keys order)
        for k in refs.keys():
            self.crefs.append(_cider_cook_refs(refs[k], n=self.n))
            if test is not None:
                self.ctest.append(_cider_cook_test(test[k][0], n=self.n))
            else:
                self.ctest.append(None)

        if doc_frequency is None and ref_len is None:
            self.compute_doc_freq()
            self.ref_len = np.log(float(len(self.crefs)))
        else:
            self.doc_frequency = doc_frequency
            self.ref_len = ref_len

    def compute_doc_freq(self):
        for refs in self.crefs:
            for ngram in set([ng for ref in refs for (ng, c) in ref.items()]):
                self.doc_frequency[ngram] += 1

    def compute_cider(self):
        def counts2vec(cnts):
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, tf) in cnts.items():
                df = np.log(max(1.0, self.doc_frequency[ngram]))
                n_idx = len(ngram) - 1
                vec[n_idx][ngram] = float(tf) * (self.ref_len - df)
                norm[n_idx] += vec[n_idx][ngram] ** 2
                if n_idx == 0:  # unigram length
                    length += tf
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            delta = float(length_hyp - length_ref)
            val = np.array([0.0 for _ in range(self.n)])
            for n_idx in range(self.n):
                for (ngram, _) in vec_hyp[n_idx].items():
                    val[n_idx] += min(vec_hyp[n_idx][ngram],
                                      vec_ref[n_idx][ngram]) * vec_ref[n_idx][ngram]
                if (norm_hyp[n_idx] != 0) and (norm_ref[n_idx] != 0):
                    val[n_idx] /= (norm_hyp[n_idx] * norm_ref[n_idx])
                val[n_idx] *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
            return val

        scores = []
        for test_cnts, refs_cnts in zip(self.ctest, self.crefs):
            vec, norm, length = counts2vec(test_cnts)
            score = np.array([0.0 for _ in range(self.n)])
            for ref_cnts in refs_cnts:
                vec_ref, norm_ref, length_ref = counts2vec(ref_cnts)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score) / len(refs_cnts) * 10.0
            scores.append(score_avg)
        return scores

    def compute_score(self):
        score = self.compute_cider()
        return np.mean(np.array(score)), np.array(score)