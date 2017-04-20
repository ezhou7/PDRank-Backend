import os
import sys
import socket

from devices import *
from pipeline import Pipeline
from properties import Properties

# from parser import PDParser
# from structure import Document, aggregate_maps
# from cluster import DocumentClustering


def main():
    path = "/Users/ezhou7/PycharmProjects/cs370/resources/cancer/breast/"
    props = Properties(indir_path=path)

    pipeline = Pipeline(props)

    kmeans_engine = pipeline.doc_clustering

    print(kmeans_engine.clusters)

    # pdf_parser = PDParser(infile_path=path)
    #
    # text = pdf_parser.parse_pdf(password="")
    # print(text)
    #
    # pdf_parser.close_parser()
    #
    # doc = Document(text)
    # aggr_map = aggregate_maps([doc.bow_map])
    # doc.vectorize(aggr_map)
    #
    # DocumentClustering.k_means(1, [doc.bow_vec])


class MainDriver:
    def __init__(self, docs_path: str=None):
        self.p_socket = None

        self._props = None
        self._pipeline = None
        self._cluster_device = None
        self._retrieval_device = None

        self._docs_path = docs_path

        self._initialize = False
        self._tear_down = False
        self._fetch_cluster = False

    def main_loop(self):
        if self.p_socket is None:
            if len(sys.argv) < 2:
                print("File descriptor not initiated.")
                return

            python_fd = int(sys.argv[1])
            self.p_socket = socket.fromfd(python_fd, socket.AF_UNIX, socket.SOCK_DGRAM)

        while True:
            cmd_num = self.p_socket.recv(4)

            if not cmd_num:
                continue

            cmd = int.from_bytes(cmd_num, sys.byteorder, signed=True)

            self.__interpret(cmd)

            if self.initialize:
                self._props = Properties(indir_path=self._docs_path)
                self._pipeline = Pipeline(self._props)

                self._cluster_device = self._pipeline.doc_clustering
                lang_model = self._pipeline.annotator.model
                self._retrieval_device = RetrievalDevice(self._cluster_device, lang_model)

            if self.tear_down:
                self.__tear_down()
                break

            if self.fetch_cluster:
                self.p_socket.send(0)
                self.__fetch_cluster()

                self._fetch_cluster = False

    def __interpret(self, command):
        if command == 0:
            # initialize
            self.initialize = True
        elif command == -1:
            # quit
            self.tear_down = True
        elif command == 1:
            # fetch cluster
            self.fetch_cluster = True

    def __tear_down(self):
        self.p_socket.shutdown()
        self.p_socket.close()

    def __fetch_cluster(self):
        buf = ""

        while True:
            user_input = self.p_socket.recv(512)
            buf += user_input.decode("utf-8")

            if len(buf) > 2048:
                break

        cluster = self._retrieval_device.fetch_cluster(buf)
        return ProcessingDevice.process_cluster(cluster)

if __name__ == "__main__":
    main()
