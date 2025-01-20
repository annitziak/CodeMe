from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Document:
    doc_id: int
    content: str


@dataclass
class Token:
    term: str
    position: int


@dataclass
class Posting:
    doc_id: int
    positions: list[int]


class PostingsList:
    def __init__(self):
        self.entries: list[Posting] = []

    def add(self, doc_id: int, position: int):
        if self.entries[-1].doc_id == doc_id:
            self.entries[-1].positions.append(position)
        else:
            self.entries.append(Posting(doc_id, [position]))

    def merge(self, other: "PostingsList"):
        self.entries.extend(other.entries)
        self.entries.sort(key=lambda x: x.doc_id)


class Indexor:
    def __init__(self, index_dir: str, shard_size: int = 1000000):
        self.index_dir = index_dir
        self.shard_size = shard_size

        self.buffer: dict[str, PostingsList] = defaultdict(PostingsList)

    def index_document(self, doc: Document):
        tokens: list[Token] = self.tokenize(doc.content)

        for token in tokens:
            self.buffer[token.term].add(doc.doc_id, token.position)

        self.buffer_docs += 1
