import logging
import os

from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PostingList:
    doc_id: int
    doc_term_frequency: int
    positions: list[int]

    def __str__(self):
        return f"{self.doc_id}:{self.doc_term_frequency}"

    def __repr__(self):
        return f"{self.doc_id}:{self.doc_term_frequency}"

    def update_position(self, position: int):
        self.positions.append(position)
        self.doc_term_frequency += 1

    def update_with_positions(self, positions: list[int]):
        for position in positions:
            self.positions.append(position)

        self.doc_term_frequency += len(positions)


@dataclass
class Term:
    term: str
    document_frequency: int
    posting_lists: dict[int, PostingList]

    def __str__(self):
        return f"{self.document_frequency}"

    def __repr__(self):
        return f"{self.document_frequency}"

    def update_with_postings(self, doc_id: int, positions: list[int]):
        if doc_id in self.posting_lists:
            self.posting_lists[doc_id].update_with_positions(positions)
            return

        self.posting_lists[doc_id] = PostingList(doc_id, len(positions), positions)

    def get_posting_list(self, doc_id: int) -> PostingList | None:
        return self.posting_lists.get(doc_id, None)


class Index:
    def __init__(self, load_path=None, index_builder_kwargs: dict = {}):
        self.load_path = load_path
        self.index_builder_kwargs = index_builder_kwargs

        self.term_map: dict[str, Term] = {}
        self.index_map = {}

        if self.load_path is not None:
            self._load_index(self.load_path)

    def __getitem__(self, term: str) -> Term:
        return self.get_term(term)

    def get_term(self, term: str) -> Term:
        term_dict = self.term_map.get(term, None)
        if term_dict is None:
            raise ValueError(f"Term {term} not found in index")

        return term_dict

    def get_document_frequency(self, term: str) -> int:
        return self.get_term(term).document_frequency

    def get_all_posting_lists(self, term: str) -> list[PostingList]:
        return list(self.get_term(term).posting_lists.values())

    def get_term_frequency(self, term: str, doc_id: int) -> int:
        posting_list = self.get_posting_list(term, doc_id)
        if posting_list is None:
            return 0

        return posting_list.doc_term_frequency

    def get_posting_list(self, term: str, doc_id: int) -> PostingList | None:
        posting_list = self.get_term(term).get_posting_list(doc_id)
        if posting_list is None:
            raise ValueError(
                f"Posting list for term {term} and doc_id {doc_id} not found"
            )

        if posting_list.doc_id != doc_id:
            raise ValueError(
                f"Posting list for term {term} and doc_id {doc_id} not found"
            )

        return posting_list

    def _load_index(self, index_path):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index path {index_path} does not exist")

        if os.path.isdir(index_path):
            with open(os.path.join(index_path, "index.txt")) as f:
                curr_term = None
                for line in f:
                    if line[0] == "\t":
                        doc_id, postings = line.strip().split("\t")
                        postings = [int(p) for p in postings.split(",")]
                        if curr_term is None:
                            raise ValueError("Invalid index format")

                        curr_term.update_with_postings(int(doc_id), postings)
                    else:
                        if curr_term is not None:
                            if curr_term.term in self.term_map:
                                raise ValueError(
                                    f"Duplicate term {curr_term.term} found in index"
                                )
                            self.term_map[curr_term.term] = curr_term

                        values = line.strip().split("\t")
                        if len(values) != 2:
                            raise ValueError("Invalid index format" + line)
                        term, document_frequency = line.strip().split("\t")
                        curr_term = Term(term, int(document_frequency), {})


if __name__ == "__main__":
    import argparse
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=str, required=True)
    args = parser.parse_args()

    index = Index(load_path=args.index_path)
    pprint.pp(index.term_map)
