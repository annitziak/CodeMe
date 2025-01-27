import logging

from indexor.structures import Term, PostingList, IndexBase
from indexor.indexor_utils import build_index


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Index(IndexBase):
    def __init__(self, load_path=None, index_builder_kwargs: dict = {}):
        self.load_path = load_path
        self.index = build_index(self.load_path, **index_builder_kwargs)

    def __getitem__(self, term: str) -> Term:
        return self.get_term(term)

    def get_term(self, term: str) -> Term:
        return self.index.get_term(term)

    def get_document_frequency(self, term: str) -> int:
        return self.index.get_document_frequency(term)

    def get_all_posting_lists(self, term: str) -> list[PostingList]:
        return self.index.get_all_posting_lists(term)

    def get_term_frequency(self, term: str, doc_id: int) -> int:
        return self.index.get_term_frequency(term, doc_id)

    def get_posting_list(self, term: str, doc_id: int) -> PostingList | None:
        return self.index.get_posting_list(term, doc_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=str, required=True)
    args = parser.parse_args()

    index = Index(load_path=args.index_path)

    while True:
        term = input("Enter a term to look up (Press q to quit): ")
        try:
            term_obj = index.get_term(term)
            print(term_obj)
        except ValueError as e:
            logger.error(e)
            continue
