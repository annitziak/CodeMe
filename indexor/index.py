import logging

from indexor.structures import Term, PostingList, IndexBase
from indexor.indexor_utils import build_index
from indexor.query import BooleanQuery, Query


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

    def get_document_length(self, doc_id: int) -> int:
        return self.index.get_document_length(doc_id)

    def get_all_posting_lists(self, term: str) -> list[PostingList]:
        return self.index.get_all_posting_lists(term)

    def get_term_frequency(self, term: str, doc_id: int) -> int:
        return self.index.get_term_frequency(term, doc_id)

    def get_posting_list(self, term: str, doc_id: int) -> PostingList | None:
        return self.index.get_posting_list(term, doc_id)

    def get_all_documents(self):
        return self.index.get_all_documents()

    def search(self, query: Query, limit: int = -1):
        if limit == -1:
            limit = 100_000

        return self.index.search(query, limit=limit)

    def get_intersection(self, terms: list[str]) -> list[int]:
        return self.index.get_intersection(terms)

    def get_union(self, terms: list[str]) -> list[int]:
        return self.index.get_union(terms)

    def get_complement(self, term: str) -> list[int]:
        return self.index.get_complement(term)

    def get_document_count(self) -> int:
        return self.index.get_document_count()

    def get_document_metadata(self, doc_id: int) -> dict:
        return self.index.get_document_metadata(doc_id)

    def get_term_count(self) -> int:
        return self.index.get_term_count()

    def write_index_to_txt(self, path: str):
        self.index.write_index_to_txt(path)

    def terms(self):
        return self.index.terms()


if __name__ == "__main__":
    import argparse
    import time
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=str, required=True)
    parser.add_argument("--write-index-to-txt", action="store_true")
    args = parser.parse_args()

    index = Index(load_path=args.index_path)
    if args.write_index_to_txt:
        index.write_index_to_txt(os.path.join(args.index_path, "index.txt"))

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
    print(f"DocumentCount={index.get_document_count()}")
    print(f"TermCount={index.get_term_count()}")

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
            elif term[:2] == "P ":
                possible_prefix = term.split(" ")
                if len(possible_prefix) == 1:
                    continue

                prefix = possible_prefix[-1]
                term_obj = index.get_term_by_prefix(prefix)
                print(f"Time taken for get_term_by_prefix: {time.time() - start}")
                print(term_obj)
            elif term[:2] == "p ":
                term = term[2:]
                term_obj = index.get_term(term, positions=True)
                print(f"Time taken for get_term: {time.time() - start}")
                print(term_obj)

                """
                start = time.time()
                term_obj = index.get_term(term, positions=True)
                print(f"Time taken for get_term with positions: {time.time() - start}")
                print(term_obj)
                """

                term_obj = index.get_posting_list(term, 15198967)
                print(f"Time taken for get_posting_list: {time.time() - start}")
                print(term_obj)
            elif term[:2] == "d ":
                doc_id = int(term[2:])
                doc_obj = index.get_document_metadata(doc_id)
                print(f"Time taken for get_document_metadata: {time.time() - start}")
                print(doc_obj)
            elif term[:2] == "s ":
                query = BooleanQuery(term[2:])
                doc_obj = index.search(query)
                print(f"Time taken for query: {time.time() - start}")
                print(doc_obj)
            else:
                term_obj = index.get_term(term, positions=False)
                print(f"Time taken for get_term: {time.time() - start}")
                print(term_obj)

        except ValueError as e:
            import traceback

            logger.error(f"Error: {e} {traceback.format_exc()}")
            continue
