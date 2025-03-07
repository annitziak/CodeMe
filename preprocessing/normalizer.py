import os
import Stemmer

from abc import ABC, abstractmethod
from preprocessing import Term
from unidecode import unidecode


class SubNormalizer(ABC):
    def normalize(self, terms: list[Term]):
        for term in terms:
            self.normalize_term(term)

        return terms

    @abstractmethod
    def normalize_term(self, term: Term):
        pass


class LowerCaseNormalizer(SubNormalizer):
    def normalize_term(self, term: Term):
        term.term = term.term.lower()


class PunctuationNormalizer(SubNormalizer):
    def normalize_term(self, term: Term):
        term.term = term.term.strip("'")


class StopWordNormalizer(SubNormalizer):
    def __init__(self, stop_words_file: str = "", stop_words_set: set[str] = set()):
        if len(stop_words_file) > 0:
            if not os.path.exists(stop_words_file):
                raise FileNotFoundError(f"Stop words file {stop_words_file} not found.")

            with open(stop_words_file, "r") as f:
                self.stop_words = set(f.read().split())
        else:
            self.stop_words = stop_words_set

    def __repr__(self):
        return f"StopWordNormalizer(stop_words={len(self.stop_words)})"

    def normalize_term(self, term: Term):
        if term.term in self.stop_words:
            term.term = ""


class StemmingNormalizer(SubNormalizer):
    def __init__(self):
        self.stemmer = Stemmer.Stemmer("english")

    def __repr__(self):
        return "StemmingNormalizer()"

    def __getstate__(self) -> object:
        data = self.__dict__.copy()
        del data["stemmer"]

        return data

    def __setstate__(self, state: object) -> None:
        self.__dict__.update(state)
        self.stemmer = Stemmer.Stemmer("english")

    def normalize_term(self, term: Term):
        term.term = self.stemmer.stemWord(term.term)


class UnicodeToAsciiNormalizer(SubNormalizer):
    """
    Normalizes the term to Normal Form such that it is decomposed into its base
    Removes accents and other diacritics
    """

    def __repr__(self):
        return "UnicodeToAsciiNormalizer()"

    def normalize_term(self, term: Term | str):
        if isinstance(term, Term):
            term.term = unidecode(term.term)
        elif isinstance(term, str):
            term = unidecode(term)

        return term


class Normalizer:
    def __init__(self, operations: list[SubNormalizer] = []):
        self.operations = operations

    def __repr__(self):
        return f"Normalizer(operations={self.operations})"

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def normalize(self, terms, filter_empty_terms=True):
        if isinstance(terms, str):
            for operation in self.operations:
                terms = operation.normalize_term(terms)
            return terms

        new_terms = []
        for term in terms:
            term.term = str(term.term).strip()
            for operation in self.operations:
                operation.normalize_term(term)
            term.term = term.term.strip()

            if filter_empty_terms and len(term.term) > 0:
                new_terms.append(term)

        return new_terms
