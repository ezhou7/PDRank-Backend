from parser import PDParser
from structure import Document, aggregate_maps
from cluster import Clustering


def main():
    path = "/Users/ezhou7/Downloads/bloomberg-proposal-2017.pdf"
    pdf_parser = PDParser(infile_path=path)

    text = pdf_parser.parse_pdf(password="")

    pdf_parser.close_parser()

    doc = Document(text)
    aggr_map = aggregate_maps([doc.bow_map])
    doc.vectorize(aggr_map)

    Clustering.k_means(doc.bow_vec)

if __name__ == "__main__":
    main()
