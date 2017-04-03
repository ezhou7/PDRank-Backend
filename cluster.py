import numpy as np
from typing import List

from scipy.cluster.vq import vq, kmeans, whiten

from structure import Document


class DocumentClustering:
    def __init__(self, docs: List[Document], k=-1):
        self.docs = docs
        self.doc_vecs = [d.bow_vec for d in docs]
        self.buffer = []
        self.clusters = None
        self.k = k if k != -1 else (7 if len(docs) >= 7 else len(docs))

        self._k_means()

    def _buf_too_buff(self):
        if len(self.buffer) > 20:
            self.docs.extend(self.buffer)
            self.doc_vecs.extend([d.bow_vec for d in self.buffer])
            self.buffer = []

            self._k_means()

    def add_doc(self, doc: Document):
        self.buffer.append(doc)
        self._buf_too_buff()

    def add_docs(self, docs: List[Document]):
        self.buffer.extend(docs)
        self._buf_too_buff()

    def _k_means(self):
        whitened = whiten(np.array(self.doc_vecs))
        code_book, _ = kmeans(whitened, self.k)
        cluster_nums, _ = vq(whitened, code_book)

        self.clusters = [[] for _ in range(len(set(cluster_nums)))]

        for i, c_num in enumerate(cluster_nums):
            self.docs[i].cluster = c_num
            self.clusters[c_num].append(self.docs[i])

    @staticmethod
    def k_means(k, vecs):
        whitened = whiten(vecs)
        code_book, _ = kmeans(whitened, k)
        cluster_num, _ = vq(whitened, code_book)
        print(cluster_num)
        return cluster_num
