import sys

from io import StringIO

from pdfminer.layout import LAParams
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter

from structure import Document


class PDParser:
    def __init__(self, infile_path: str=None, outfile_path: str=None):
        self.infile_path = infile_path
        self.outfile_path = outfile_path

        self.fin = None
        self.fout = StringIO()
        self._is_str_output = True

        self.la_params = LAParams()
        self.resrc_mgr = PDFResourceManager()
        self.pdf_parser = None

        if self.infile_path is not None:
            self.open_file(self.infile_path, "i")

        if self.outfile_path is not None:
            self.open_file(self.outfile_path, "o")

    def open_file(self, file_path: str=None, io_func: str="i"):
        if file_path is not None:
            if io_func == "i":
                fin = open(file_path, "rb")
                self.pdf_parser = PDFParser(fin)

                self.infile_path = file_path
                self.fin = fin
            elif io_func == "o":
                fout = open(file_path, "w")

                self.outfile_path = file_path
                self.fout = fout

        else:
            print("Please fill in the file input parameter.")

    def attach_fstream(self, fstream=None, io_func: str="i"):
        if fstream is None:
            print("Please fill in the file stream input parameter.")
            return

        if io_func == "i":
            self.fin = fstream
        elif io_func == "o":
            self.fout = fstream

    def close_parser(self):
        if self.fin is not None:
            self.fin.close()

        if self.fout is not sys.stdout or self.fout is not None:
            self.fout.close()

        if self.pdf_parser is not None:
            self.pdf_parser.close()

    def parse_pdf(self, password: str=""):
        if self.fin is None:
            print("PDParse class - parse_pdf() Error:")
            print("Fatal: No pdf file provided.")
            return None
        else:
            if self.pdf_parser is None:
                self.pdf_parser = PDFParser(self.fin)

        pdf_doc = PDFDocument(self.pdf_parser, password=password)

        if not pdf_doc.is_extractable:
            raise PDFTextExtractionNotAllowed

        pdf_device = TextConverter(self.resrc_mgr, self.fout, laparams=self.la_params)
        pdf_interpreter = PDFPageInterpreter(self.resrc_mgr, pdf_device)

        for page in PDFPage.create_pages(document=pdf_doc):
            pdf_interpreter.process_page(page)

        pdf_device.close()

        if isinstance(self.fout, StringIO):
            return Document(text=self.fout.getvalue())
        else:
            return None
