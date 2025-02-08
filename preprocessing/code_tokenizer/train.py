import logging
import re

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import (
    PreTokenizer,
    Whitespace,
    Sequence as PreTokenizerSequence,
)
from tokenizers.normalizers import (
    NFD,
    StripAccents,
    Strip,
    Sequence,
    Lowercase,
)


from data.code_search_net import CodeSearchNetDataset, CodeDataset
from preprocessing.parser import HTMLParserInterface as HTMLParser

logger = logging.getLogger(__name__)


class CustomPreTokenizer:
    def pre_tokenize(self, text):
        text = re.sub(r"([^\s\w\[\]{}<>().,;:=+\-!*_/])", r" \1 ", text)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"([A-Z])([a-z])", r"\1 \2", text)
        text = re.sub(r"([a-zA-Z]+)(\d+)", r"\1 \2", text)
        text = re.sub(r"(\d+)([a-zA-Z]+)", r"\1 \2", text)

        text = re.sub(r"\s+", " ", text)

        return [(word, (0, 0)) for word in text.split()]


class BPETokenizer:
    def __init__(self, vocab_size: int, save_path: str):
        self.tokenizer = Tokenizer(BPE())

        self.tokenizer.normalizer = Sequence(
            [NFD(), StripAccents(), Strip(), Lowercase()]
        )

        custom_pre_tokenizer = PreTokenizer.custom(CustomPreTokenizer())
        self.tokenizer.pre_tokenizer = PreTokenizerSequence(
            [custom_pre_tokenizer, Whitespace()]
        )

        self.vocab_size = vocab_size

        self.save_path = save_path

    def save(self):
        self.tokenizer.save(self.save_path)

    def load(self):
        self.tokenizer = Tokenizer.from_file(self.save_path)
        return self

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)


class TokenizerTrainer:
    def __init__(self, tokenizer: BPETokenizer):
        self.tokenizer = tokenizer
        self.trainer = BpeTrainer(
            vocab_size=self.tokenizer.vocab_size, special_tokens=["<UNK>"]
        )

    def train(self, iterator):
        self.tokenizer.tokenizer.train_from_iterator(iterator, self.trainer)
        self.tokenizer.save()


class SQLDataset:
    def __init__(self, db_conn, number_of_samples: int, batch_size: int):
        self.db_conn = db_conn
        self.html_parser = HTMLParser()

        self.number_of_samples = number_of_samples
        self.batch_size = batch_size

        self.fetched = 0

    def __len__(self):
        return self.number_of_samples

    def load(self):
        select_query = f"SELECT id,title,body FROM posts WHERE body IS NOT NULL AND body != '' LIMIT {self.number_of_samples}"

        with self.db_conn as cursor:
            cursor.execute(select_query, commit=False)
            while self.fetched < self.number_of_samples:
                batch = cursor.fetchmany(self.batch_size)
                self.fetch_next = False

                texts = []
                for row in batch:
                    text = row[1] + " " + self.html_parser.parse(row[2])
                    texts.append(text)
                    self.fetched += 1

                yield texts


class DatasetIterator:
    def __init__(self, dataset: CodeDataset, sql_dataset: SQLDataset, batch_size: int):
        self.batch_size = batch_size
        self.dataset = dataset

        self.sql_dataset = sql_dataset
        self.iter_sql_dataset = iter(sql_dataset.load())

    def __len__(self):
        return len(self.dataset) + len(self.sql_dataset)

    def __iter__(self):
        for i in range(0, len(self), self.batch_size):
            if i < len(self.dataset):
                yield self.dataset[i : i + self.batch_size]
            else:
                yield next(self.iter_sql_dataset)


if __name__ == "__main__":
    from constants.db import DB_PARAMS
    from utils.db_connection import DBConnection

    db_conn = DBConnection(DB_PARAMS)

    dataset = CodeSearchNetDataset()
    sql_dataset = SQLDataset(db_conn, 1000, 32)

    tokenizer = BPETokenizer(50256, ".cache/tokenizer.json")
    trainer = TokenizerTrainer(tokenizer)
    iterator = DatasetIterator(dataset, sql_dataset, 32)

    trainer.train(iterator)
