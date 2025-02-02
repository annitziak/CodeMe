import logging

from indexor.structures import Term, PostingList, IndexBase
from indexor.indexor_utils import build_index


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Index(IndexBase):
    def __init__(
        self, load_path=None, use_disk_index=True, index_builder_kwargs: dict = {}
    ):
        self.load_path = load_path
        self.use_disk_index = use_disk_index
        self.index = build_index(
            self.load_path, self.use_disk_index, **index_builder_kwargs
        )

    def __getitem__(self, term: str) -> Term:
        return self.get_term(term)

    def get_term(self, term: str, positions=False) -> Term:
        return self.index.get_term(term, positions=positions)

    def get_term_by_prefix(self, prefix: str) -> list[str]:
        return self.index.get_term_by_prefix(prefix)

    def get_document_frequency(self, term: str) -> int:
        return self.index.get_document_frequency(term)

    def get_all_posting_lists(self, term: str) -> list[PostingList]:
        return self.index.get_all_posting_lists(term)

    def get_term_frequency(self, term: str, doc_id: int) -> int:
        return self.index.get_term_frequency(term, doc_id)

    def get_posting_list(self, term: str, doc_id: int) -> PostingList | None:
        return self.index.get_posting_list(term, doc_id)

    def get_all_documents(self):
        return self.index.get_all_documents()

    def get_intersection(self, terms: list[str]) -> list[int]:
        return self.index.get_intersection(terms)

    def get_union(self, terms: list[str]) -> list[int]:
        return self.index.get_union(terms)

    def get_complement(self, term: str) -> list[int]:
        return self.index.get_complement(term)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=str, required=True)
    args = parser.parse_args()

    index = Index(load_path=args.index_path)

    """
    term_obj = index.get_term("!", positions=False)
    print(term_obj)

    term_obj = index.get_union(["hello", "python"])
    print(term_obj)

    term_obj = index.get_intersection(["hello", "python"])
    print(term_obj)
    """

    """
    terms = index.get_all_documents()
    print(terms)

    with open("document_terms.txt", "w") as f:
        for doc_id, terms in terms.items():
            term_str = "\t".join(terms)
            f.write(f"{doc_id} : {term_str}")
            f.write("\n")
    """

    while True:
        term = input(
            "Enter a term to look up (Press q to quit) or a query decided by the first two characters 'BB'; use 'P' to search by prefix:"
        )
        start = time.time()
        try:
            if term[:2] == "BB":
                terms = term.split(" ")[1:]
                if len(terms) == 1 and terms[0][0] == "!":
                    term_obj = index.get_complement(terms[1:])
                    print(f"Time taken for get_complement: {time.time() - start}")
                    print(term_obj)
                elif len(terms) == 3 and terms[1] == "<I>":
                    term_obj = index.get_intersection([terms[0], terms[2]])
                    print(f"Time taken for get_intersection: {time.time() - start}")
                    print(term_obj)
                elif len(terms) == 3 and terms[1] == "<U>":
                    term_obj = index.get_union([terms[0], terms[2]])
                    print(f"Time taken for get_union: {time.time() - start}")
                    print(term_obj)
                else:
                    print(f"Invalid query: {terms}")
            elif term[:1] == "P":
                prefix = term.split(" ")[1]
                term_obj = index.get_term_by_prefix(prefix)
                print(f"Time taken for get_term_by_prefix: {time.time() - start}")
                print(term_obj)
            else:
                term_obj = index.get_term(term, positions=True)
                print(f"Time taken for get_term: {time.time() - start}")
                print(term_obj)

                term_obj = index.get_posting_list(term, 15198967)
                print(f"Time taken for get_posting_list: {time.time() - start}")
                print(term_obj)
        except ValueError as e:
            logger.error(e)
            continue
