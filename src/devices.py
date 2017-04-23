import numpy as np
from structure import Document


class Processor:
    def __init__(self, clusterer=None, annotator=None):
        self.clusterer = clusterer
        self.annotator = annotator

    def fetch_cluster(self, search_input):
        input_doc = Document(search_input)
        return self._get_max_cluster(input_doc)

    def _compare(self, doc, cluster):
        d_vec = self._standardize(doc)
        return np.dot(d_vec, cluster.aggr_vec)

    def _standardize(self, doc):
        w2i = self.clusterer.w2i_map
        diff = len(set(doc.bow_map.keys()) - set(w2i.keys()))
        size = self.clusterer.counter + diff

        d_array = np.zeros(size)

        for key in doc.bow_map:
            if key in w2i:
                d_array[w2i[key]] = doc.bow_map[key]

        return d_array

    def _get_max_cluster(self, doc):
        max_cidx = np.argmax(np.array([self._compare(doc, cluster) for cluster in self.clusterer.clusters]))
        return self.clusterer.clusters[max_cidx]
