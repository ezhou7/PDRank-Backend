import numpy as np
from structure import Document


class RetrievalDevice:
    def __init__(self, cluster_device=None, lang_model=None):
        self.cluster_device = cluster_device
        self.lang_model = lang_model

    def fetch_cluster(self, search_input):
        input_doc = Document(search_input)
        return self._get_max_cluster(input_doc)

    def _compare(self, doc, cluster):
        return np.dot(doc.bow_vec, cluster.aggr_vec)

    def _get_max_cluster(self, doc):
        max_cidx = np.argmax(np.array([self._compare(doc, cluster) for cluster in self.cluster_device.clusters]))
        return self.cluster_device.clusters[max_cidx]


class ProcessingDevice:
    @staticmethod
    def process_cluster(cluster):
        """
        Returns a list of titles
        :param cluster: list of documents
        :return: list of pdf titles
        """
        return [doc.title for doc in cluster.docs]
