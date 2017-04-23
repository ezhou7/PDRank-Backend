import re
import spacy
from typing import List, Tuple
from collections import Counter

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
        return [t.lemma_ for t in self.model(doc.text) if not t.is_stop]

    def doc_to_bow(self, doc: Document):
        lemmas = self.lemmatize_doc(doc)
        doc.bow_map = Counter(lemmas)

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

    def extract_date(self, doc: Document):
        doc.date = self._extract_max_date(doc)

    def _extract_max_date(self, doc: Document) -> Tuple[int, int, int]:
        # -- Compile different forms of dates into regex objects --

        # year only (i.e. "2017")
        year = re.compile("(20\\d{2}|19\\d{2})")

        # US date style (i.e. "Jan. 1, 2017")
        us_full_date = re.compile("(\\w+ \\d{1,2},? \\d{4})")

        # numerical date style (i.e. "1/1/2017")
        num_date = re.compile("(\\d{1,2}/\\d{1,2}/\\d{2,4})")

        # numerical date style with no day (i.e. "1/2017")
        num_date_no_day = re.compile("(\\d{1,2}/\\d{2,4})")

        # -- Find all instances of regex matches --

        year_matches = year.findall(doc.text)

        us_full_date_matches = us_full_date.findall(doc.text)

        num_date_matches = num_date.findall(doc.text)

        num_date_no_day_matches = num_date_no_day.findall(doc.text)

        # -- Find max date --

        # standardized date primitive: tuple -> (year: int, month: int, day: int)
        max_year = (max(map(int, year_matches)) if len(year_matches) > 0 else -1, -1, -1)

        # convert US date structure to standardized date structure
        us_dates = self._standardize_dates(us_full_date_matches, date_type="usa")
        max_us_date = max(us_dates) if len(us_dates) > 0 else (-1, -1, -1)

        num_dates = self._standardize_dates(num_date_matches, date_type="num")
        num_date_max = max(num_dates) if len(num_dates) > 0 else (-1, -1, -1)

        num_dates_incomp = self._standardize_dates(num_date_no_day_matches, date_type="num_inc")
        num_date_incomp_max = max(num_dates_incomp) if len(num_dates_incomp) > 0 else (-1, -1, -1)

        return max((max_year, max_us_date, num_date_max, num_date_incomp_max))

    def _split_dates(self, dates, date_type="usa"):
        regex = None

        if date_type == "usa":
            regex = "[,]? "
        if date_type == "num" or date_type == "num_inc":
            regex = "/"

        return [re.split(regex, d) for d in dates]

    def _standardize_dates(self, dates, date_type="usa"):
        split_dates = self._split_dates(dates, date_type=date_type)
        return [self._standardize_date(d, date_type) for d in split_dates]

    def _standardize_date(self, date, date_type="usa"):
        std_date = None

        if date_type == "usa":
            std_date = (int(date[2]), self._convert_month(date[0]), int(date[1]))
        elif date_type == "num":
            std_date = (int(date[2]), int(date[1]), int(date[0]))
        elif date_type == "num_inc":
            std_date = (int(date[1]), int(date[0]), -1)

        return std_date

    def _convert_month(self, month: str) -> int:
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
