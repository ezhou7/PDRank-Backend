import re
import spacy

from typing import List, Tuple

from structure import Document


class Annotator:
    def __init__(self, language: str="en"):
        self.language = language
        self.model = spacy.load(language)

    def fill_out_doc(self, doc: Document):
        self.doc_to_bow(doc)
        self.extract_title(doc)
        self.extract_date(doc)

    def lemmatize_doc(self, doc: Document) -> List[str]:
        return [token.lemma_ if not token.is_stop else None for token in self.model(doc.text)]

    def doc_to_bow(self, doc: Document):
        lemmas = self.lemmatize_doc(doc)

        if not doc.bow_map:
            doc.bow_map = dict()

        for lemma in lemmas:
            if lemma is not None:
                stripped = lemma.strip()

                if stripped not in doc.bow_map:
                    doc.bow_map[stripped] = 1
                else:
                    doc.bow_map[stripped] += 1

    def extract_title(self, doc: Document):
        not_title = re.compile("\\W+")

        lines = doc.text.split("\n")

        para_sz = 0
        para = None
        for i, line in enumerate(lines):
            if line == "" and para_sz == 0:
                continue
            elif line == "" and para_sz != 0:
                para = lines[i - para_sz:i]
                break

            not_title_match = not_title.match(line)
            if not_title_match:
                para_sz = 0
                continue

            para_sz += 1

        doc.title = " ".join(para) if para is not None else ""
        print(doc.title)

    def extract_date(self, doc: Document):
        doc.date = self._extract_max_date(doc)

    def _extract_max_date(self, doc: Document) -> Tuple[int, int, int]:
        # -- Compile different forms of dates into regex objects --

        # year only (i.e. "2017")
        year = re.compile("(20\\d{2}|19\\d{2})")

        # US date style (i.e. "Jan. 1, 2017")
        us_full_date = re.compile("(\\w+ \\d{1,2},? \\d{4})")

        # STD date style (i.e. "1 Jan., 2017")
        # eu_full_date = re.compile("(\\d{1,2} \\w+,? \\d{4})")

        # Only month and year (i.e. "Jan. 2017")
        # no_day = re.compile("(\\w+,? \\d{4})")

        # numerical date style (i.e. "1/1/2017")
        num_date = re.compile("(\\d{1,2}/\\d{1,2}/\\d{2,4})")

        # numerical date style with no day (i.e. "1/2017")
        num_date_no_day = re.compile("(\\d{1,2}/\\d{2,4})")

        # -- Find all instances of regex matches --

        year_matches = year.findall(doc.text)

        us_full_date_matches = us_full_date.findall(doc.text)
        # eu_full_date_matches = eu_full_date.findall(doc.text)

        # no_day_matches = no_day.findall(doc.text)

        num_date_matches = num_date.findall(doc.text)
        num_date_no_day_matches = num_date_no_day.findall(doc.text)

        # -- Find max date --

        # standardized date primitive: tuple -> (year: int, month: int, day: int)
        max_year = (max(map(int, year_matches)) if len(year_matches) > 0 else -1, -1, -1)

        # convert US date structure to standardized date structure
        us_dates_split = [re.split("[,]? ", d) for d in us_full_date_matches]
        us_dates = [(int(d[2]), month_converter(d[0]), int(d[1])) for d in us_dates_split]
        max_us_date = max(us_dates) if len(us_dates) > 0 else (-1, -1, -1)

        # eu_dates_split = [re.split("[,]? ", d) for d in eu_full_date_matches]
        # print("EU date: " + str(eu_dates_split))
        # eu_dates = [(int(d[2]), month_converter(d[1]), int(d[0])) for d in eu_dates_split]
        # max_eu_date = max(eu_dates) if len(eu_dates) > 0 else -1

        # no_day_split = [re.split("[,]? ", d) for d in no_day_matches]
        # print("No day date: " + str(no_day_split))
        # no_day_dates = [(int(d[1]), int(d[0]), -1) for d in no_day_split]
        # max_no_day_date = max(no_day_dates) if len(no_day_dates) > 0 else -1

        num_date_split = [d.split("/") for d in num_date_matches]
        num_dates = [(int(d[2]), int(d[1]), int(d[0])) for d in num_date_split]
        num_date_max = max(num_dates) if len(num_dates) > 0 else (-1, -1, -1)

        num_date_no_day_split = [d.split("/") for d in num_date_no_day_matches]
        num_dates_incomp = [(int(d[1]), int(d[0]), -1) for d in num_date_no_day_split]
        num_date_incomp_max = max(num_dates_incomp) if len(num_dates_incomp) > 0 else (-1, -1, -1)

        return max((max_year, max_us_date, num_date_max, num_date_incomp_max))


def month_converter(month: str) -> int:
    m_lowered = month.lower()
    if m_lowered in ["jan", "jan.", "january"]:
        return 1
    elif m_lowered in ["feb", "feb.", "february"]:
        return 2
    elif m_lowered in ["mar", "mar.", "march"]:
        return 3
    elif m_lowered in ["apr", "apr.", "april"]:
        return 4
    elif m_lowered == "may":
        return 5
    elif m_lowered in ["jun", "jun.", "june"]:
        return 6
    elif m_lowered in ["jul", "jul.", "july"]:
        return 7
    elif m_lowered in ["aug", "aug.", "august"]:
        return 8
    elif m_lowered in ["sept", "sept.", "september"]:
        return 9
    elif m_lowered in ["oct", "oct.", "october"]:
        return 10
    elif m_lowered in ["nov", "nov.", "november"]:
        return 11
    elif m_lowered in ["dec", "dec.", "december"]:
        return 12
    else:
        return -1
