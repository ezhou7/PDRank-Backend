from structure import Document, aggregate_maps
from cluster import DocumentClustering
from parser import PDParser

def main():
    path = r"C:\Users\Chris\Documents\Christopher\Research\Malaria Research\General\The pathogenic basis of malaria.pdf"
    pdf_parser = PDParser(infile_path=path)

    text = pdf_parser.parse_pdf(password="")
    print(text)

    pdf_parser.close_parser()

    doc = Document(text)
    aggr_map = aggregate_maps([doc.bow_map])
    doc.vectorize(aggr_map)

    DocumentClustering.k_means(1, [doc.bow_vec])


def main_loop():
    while True:
        # TODO: Listen to requests from rails server
        pass

if __name__ == "__main__":
    main()
