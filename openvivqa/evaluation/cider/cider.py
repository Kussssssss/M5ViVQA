# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>

from .cider_scorer import CiderScorer

class Cider:
    """Main CIDEr wrapper with cached document frequency."""
    def __init__(self, gts=None, n: int = 4, sigma: float = 6.0):
        self._n = int(n)
        self._sigma = float(sigma)
        self.doc_frequency = None
        self.ref_len = None
        if gts is not None:
            tmp = CiderScorer(gts, n=self._n, sigma=self._sigma)
            self.doc_frequency = tmp.doc_frequency
            self.ref_len = tmp.ref_len

    def compute_score(self, gts, res):
        # gts/res: Dict[id, List[str]]
        assert gts.keys() == res.keys()
        scorer = CiderScorer(
            gts, test=res, n=self._n, sigma=self._sigma,
            doc_frequency=self.doc_frequency, ref_len=self.ref_len
        )
        return scorer.compute_score()

    def __str__(self):
        return "CIDEr"
