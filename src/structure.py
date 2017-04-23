import spacy
import numpy as np
from enum import IntEnum
from collections import Counter


class DateIndex(IntEnum):
    day = 2
    month = 1
    year = 0


class Document:
    def __init__(self, text=""):
        self.text = text
        self.bow_map = None
        self.bow_vec = None
        self.title = None
        self.date = None
        self.cluster = None

    def __str__(self):
        date = "%d/%d/%d" % (self.date[DateIndex.month], self.date[DateIndex.day], self.date[DateIndex.year])
        return "Title: %s, Date: %s" % (self.title, date)

    def import_text(self, text):
        self.text = text

    def import_file(self, file_path):
        fin = open(file_path, "r")
        self.text = fin.read()
        fin.close()

    def import_ifstream(self, ifstream):
        self.text = ifstream.read()

    def _to_bag_of_words(self):
        eng_model = spacy.load("en")
        doc = eng_model(self.text)

        self.bow_map = Counter([t.lemma_.strip() for t in doc if not t.is_stop])

    def vectorize(self, word_index_map):
        self.bow_vec = np.zeros(shape=(len(word_index_map),))

        for lemma in self.bow_map.keys():
            self.bow_vec[word_index_map[lemma]] = self.bow_map[lemma]


class Cluster:
    def __init__(self, c_id=-1, docs=None):
        self.c_id = c_id
        self.docs = docs
        self.aggr_vec = self._aggregate()

    def __str__(self):
        return "Cluster ID: %d, Cluster Docs: %s" % (self.c_id, self._stringify_docs())

    def _stringify_docs(self):
        if len(self.docs) > 10:
            docs_head = ", ".join([d.title for d in self.docs[:3]])
            docs_tail = ", ".join([d.title for d in self.docs[-3:]])

            return "[%s, ..., %s]" % (docs_head, docs_tail)
        else:
            return str([str(d) for d in self.docs])

    def _aggregate(self):
        if len(self.docs) == 0:
            return np.zeros(shape=self.docs[0].bow_vec.shape)

        doc_stack = np.vstack([doc.bow_vec for doc in self.docs])

        return np.sum(doc_stack, axis=0) # , sum([d.bow_map for d in self.docs], Counter())
