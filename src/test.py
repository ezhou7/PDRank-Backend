import os

from parser import PDParser
from structure import Document


def test_path(path=""):
    is_valid = os.path.exists(path)
    assert is_valid, "Path is invalid!"

    return is_valid


def test_document():
    doc = Document()
    assert doc.text == "", "Document not properly created!"


def test_parser():
    path = "../resources/brain_002.pdf"

    if not test_path(path):
        return

    parser = PDParser(infile_path=path)

    pdf_doc = parser.parse_pdf(password="")

    parser.close_parser()
