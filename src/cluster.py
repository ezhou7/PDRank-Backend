import numpy as np
from typing import List

from scipy.cluster.vq import vq, kmeans, whiten

from structure import Document


class DocumentClustering:
    def __init__(self, docs: List[Document], k: int=-1):
        self.docs = docs
        self.doc_vecs = None
        self.buffer = []
        self.clusters = None
        self.k = k if k != -1 else (7 if len(docs) >= 7 else len(docs))

        self.w2i_map = {}
        self.counter = 0

        self._update_state()

    def _buf_too_buff(self):
        if len(self.buffer) > 20:
            self.docs.extend(self.buffer)
            self.doc_vecs.extend([d.bow_vec for d in self.buffer])

            self.buffer = []

            self._standardize()
            self._k_means()

    def _standardize(self):
        for doc in self.docs:
            for word in doc.bow_map.keys():
                if word not in self.w2i_map:
                    self.w2i_map[word] = self.counter
                    self.counter += 1

        for d in self.docs:
            d.vectorize(self.w2i_map)

        self.doc_vecs = np.array([d.bow_vec for d in self.docs])

    def _update_state(self):
        self._standardize()
        self._k_means()

    def add_doc(self, doc: Document):
        self.buffer.append(doc)
        self._buf_too_buff()

    def add_docs(self, docs: List[Document]):
        self.buffer.extend(docs)
        self._buf_too_buff()

    def _k_means(self):
        whitened = whiten(self.doc_vecs)
        code_book, _ = kmeans(whitened, self.k)
        cluster_nums, _ = vq(whitened, code_book)

        self.clusters = [[] for _ in range(len(set(cluster_nums)))]

        for i, c_num in enumerate(cluster_nums):
            self.docs[i].cluster = c_num
            self.clusters[c_num].append(self.docs[i])
