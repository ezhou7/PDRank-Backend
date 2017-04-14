import os
from typing import List

from parser import PDParser
from structure import Document
from annotation import Annotator
from properties import Properties
from cluster import DocumentClustering


class Pipeline:
    def __init__(self, props: Properties=Properties()):
        self.props = props
        self.parser = None
        self.annotation = None
        self.doc_clustering = None

        docs = None
        if props.parse:
            docs = self._parse_files(props.indir_path)

        if props.annotate:
            self._annotate_docs(docs)

        if props.cluster:
            self._cluster_docs(docs)

    def _parse_file(self, infile_path: str):
        self.parser = PDParser(infile_path=infile_path)
        doc = self.parser.parse_pdf(password="")
        self.parser.close_parser()

        return doc

    def _parse_files(self, indir_path: str):
        return [self._parse_file(indir_path + f) for f in os.listdir(indir_path)]

    def _annotate_doc(self, doc: Document):
        if self.annotation is None:
            self.annotation = Annotator()

        self.annotation.fill_out_doc(doc)

    def _annotate_docs(self, docs: List[Document]):
        for doc in docs:
            self._annotate_doc(doc)

    def _cluster_docs(self, docs: List[Document]):
        if self.doc_clustering is None:
            self.doc_clustering = DocumentClustering(docs)
        else:
            self.doc_clustering.add_docs(docs)
