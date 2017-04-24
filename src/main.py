import os

from devices import Processor
from pipeline import Pipeline
from properties import Properties


def main():
    path = "/Users/ezhou7/PycharmProjects/cs370/resources/cancer/breast/"
    engine = MainDriver(docs_path=path)
    engine.main_loop()


class MainDriver:
    def __init__(self, docs_path: str=None):
        self._props = Properties(indir_path=docs_path)
        self._pipeline = Pipeline(self._props)
        self._processor = Processor(clusterer=self._pipeline.doc_clustering, annotator=self._pipeline.annotator)

        self._docs_path = docs_path

        self._input_toggle = "/Users/ezhou7/PycharmProjects/cs370/resources/input.txt"
        self._output_toggle = "/Users/ezhou7/PycharmProjects/cs370/resources/output.txt"

        self._input_path = "/Users/ezhou7/PycharmProjects/cs370/resources/user_input.txt"
        self._output_path = "/Users/ezhou7/PycharmProjects/cs370/resources/prog_output.txt"

        self._exit_toggle = "/Users/ezhou7/PycharmProjects/cs370/resources/exit.txt"

    def _input_exists(self):
        return os.path.exists(self._input_toggle)

    def _read_input(self):
        fin = open(self._input_path, "r")

        buf = []
        for line in fin:
            buf.append(line.strip())

        data = "".join(buf)

        fin.close()

        return data

    def _process_search(self):
        user_input = self._read_input()

        if user_input:
            c = self._processor.fetch_cluster(user_input)

            if c:
                fout = open(self._output_path, "w")

                for doc in c.docs:
                    fout.write("%s\n" % doc.doc_path)

                fout.close()

                fout = open(self._output_toggle, "w")
                fout.close()

            os.remove(self._input_path)
            os.remove(self._input_toggle)

    def _quit(self):
        return os.path.exists(self._exit_toggle)

    def main_loop(self):
        print("Ready")

        while True:
            if self._quit():
                os.remove(self._exit_toggle)
                break

            if self._input_exists():
                self._process_search()

if __name__ == "__main__":
    main()
