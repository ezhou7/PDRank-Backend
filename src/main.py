import os

from devices import *
from pipeline import Pipeline
from properties import Properties


def main():
    path = "/Users/ezhou7/PycharmProjects/cs370/resources/cancer/breast/"
    props = Properties(indir_path=path)

    pipeline = Pipeline(props)

    kmeans_engine = pipeline.doc_clustering

    print([str(c) for c in kmeans_engine.clusters])


class MainDriver:
    def __init__(self, docs_path: str=None):
        self._props = None
        self._pipeline = None

        self._docs_path = docs_path

        self.input_path = "/Users/ezhou7/PycharmProjects/cs370/resources/user_input.txt"
        self.output_path = "/Users/ezhou7/PycharmProjects/cs370/resources/output.txt"

    def _input_exists(self):
        return os.path.exists(self.input_path)

    def _read_input(self):
        fin = open(self.input_path, "r")
        data = "".join([line for line in fin])
        fin.close()

        return data

    def _process_search(self):
        user_input = self._read_input()
        if user_input:
            pass

        os.remove(self.input_path)

    def main_loop(self):
        while True:
            if self._input_exists():
                self._process_search()

if __name__ == "__main__":
    # fin = open("/Users/ezhou7/PycharmProjects/cs370/resources/user_input.txt", "r")
    #
    # buf = []
    #
    # for line in fin:
    #     buf.append(line)
    #
    # data = "".join(buf)
    # print(data)
    #
    # fin.close()

    main()
