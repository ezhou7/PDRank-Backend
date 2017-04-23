import numpy as np
from structure import Document
from collections import Counter


class Processor:
    def __init__(self, clusterer=None, annotator=None):
        self.clusterer = clusterer
        self.annotator = annotator

    def fetch_cluster(self, search_input):
        input_doc = Document(search_input)
        return self._get_max_cluster(input_doc)

    def _compare(self, doc, cluster):
        d_vec, c_vec = self._standardize(doc, cluster)
        return np.dot(d_vec, c_vec)

    def _standardize(self, doc, cluster):
        self.annotator.doc_to_bow(doc)
        w2i = self.clusterer.w2i_map
        # for w in self.clusterer.w2i_map:
        #     w2i[w] = self.clusterer.w2i_map[w]
        #
        # i = len(w2i)
        # for w in doc.bow_map:
        #     if not w in w2i:
        #         w2i[w] = i
        #         i += 1

        diff = len(set(doc.bow_map.keys()) - set(w2i.keys()))

        size = len(w2i) + diff

        d_array = np.zeros(size)

        for key in doc.bow_map:
            d_array[w2i[key]] = doc.bow_map[key]

        c_array = np.zeros(size)
        c_array[:len(cluster.aggr_vec)] = cluster.aggr_vec

        return d_array, c_array

    def _get_max_cluster(self, doc):
        max_cidx = np.argmax(np.array([self._compare(doc, cluster) for cluster in self.clusterer.clusters]))
        return self.clusterer.clusters[max_cidx]
