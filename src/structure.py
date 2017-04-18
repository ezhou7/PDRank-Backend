import spacy
import numpy as np
import pickle


class Document:
    def __init__(self, text=""):
        self.text = text
        self.bow_map = None
        self.bow_vec = None
        self.title = None
        self.date = None
        self.cluster = None

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

        self.bow_map = dict()

        for token in doc:
            if not token.is_stop:
                stripped = token.lemma_.strip()
                if token.lemma_ not in self.bow_map:
                    self.bow_map[stripped] = 1
                else:
                    self.bow_map[stripped] += 1

    def vectorize(self, word_index_map):
        self.bow_vec = np.zeros(shape=(len(word_index_map),))

        for lemma in self.bow_map.keys():
            self.bow_vec[word_index_map[lemma]] = self.bow_map[lemma]


class Cluster:
    def __init__(self, docs=None):
        self.docs = docs
        self.aggr_vec = self._aggregate()

    def _aggregate(self):
        if len(self.docs) == 0:
            return np.zeros(shape=self.docs[0].bow_vec.shape)

        doc_stack = np.vstack([doc.bow_vec for doc in self.docs])
        return np.sum(doc_stack, axis=0)


class PersistentIndexMap:
    @staticmethod
    def save_map(obj_map, obj_out):
        pickle.dump(obj_map, obj_out)

    @staticmethod
    def load_map(obj_path):
        return pickle.load(obj_path)


def aggregate_maps(maps):
    aggr_map = dict()

    i = 0
    for map in maps:
        for key in map.keys():
            if key not in aggr_map:
                aggr_map[key] = i
                i += 1

    return aggr_map
