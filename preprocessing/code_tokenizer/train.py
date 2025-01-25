from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split, ByteLevel
from tokenizers.normalizers import NFD, StripAccents, Strip, Sequence


from data.code_search_net import CodeSearchNetDataset, CodeDataset


class PreTokenizer:
    def __init__(self, *args):
        self.pre_tokenizers = args

    def __call__(self, text):
        for pre_tokenizer in self.pre_tokenizers:
            text, _ = pre_tokenizer(text)
        return text, None


class CodeTokenizer:
    def __init__(self, vocab_size: int, save_path: str):
        self.tokenizer = Tokenizer(BPE())

        self.tokenizer.normalizer = Sequence([NFD(), StripAccents(), Strip()])
        self.tokenizer.pre_tokenizer = Split(
            pattern=Regex(r"[^\S\r\n]+|[{}()\[\],.;]"), behavior="isolated"
        )
        self.tokenizer.pre_tokenizer = ByteLevel()

        self.vocab_size = vocab_size

        self.save_path = save_path

    def save(self):
        self.tokenizer.save(self.save_path)

    def load(self):
        self.tokenizer = Tokenizer.from_file(self.save_path)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)


class TokenizerTrainer:
    def __init__(self, tokenizer: CodeTokenizer):
        self.tokenizer = tokenizer
        self.trainer = BpeTrainer(
            vocab_size=self.tokenizer.vocab_size, special_tokens=["<UNK>"]
        )

    def train(self, iterator):
        self.tokenizer.tokenizer.train_from_iterator(iterator, self.trainer)
        self.tokenizer.save()


class DatasetIterator:
    def __init__(self, dataset: CodeDataset, batch_size: int):
        self.batch_size = batch_size
        self.dataset = dataset

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            yield self.dataset[i : i + self.batch_size]


if __name__ == "__main__":
    dataset = CodeSearchNetDataset()
    tokenizer = CodeTokenizer(50256, ".cache/tokenizer.json")
    trainer = TokenizerTrainer(tokenizer)
    iterator = DatasetIterator(dataset, 32)

    trainer.train(iterator)
