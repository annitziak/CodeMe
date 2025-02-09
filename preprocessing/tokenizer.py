import re
import logging

from dataclasses import dataclass
from preprocessing import CodeBlock, NormalTextBlock as TextBlock, LinkBlock, Term
from preprocessing.normalizer import Normalizer
from preprocessing.code_tokenizer.train import BPETokenizer

from tokenizers.pre_tokenizers import (
    PreTokenizer,
    Whitespace,
    Sequence,
)
from preprocessing.code_tokenizer.train import CustomPreTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TokenizedOutput:
    tokenized_text: list[Term]
    original_number_of_words: int = 0


class CodeTokenizer:
    def __init__(self, use_bpe=True, bpe_length_threshold=15):
        """
        Naive tokenizer for code snippets

        Based on keywords and operators found
        """
        self.camel_case_re = re.compile(r"([a-z])([A-Z])|([A-Z])([A-Z][a-z])")
        # self.symbol_re = re.compile(r"([.,;:(){}\[\]=+\-*/])")
        self.symbol_re = re.compile(r"(\d+|[.,;:(){}\[\]=+\-*$<>?#!~@^&\|/]|\w+)")

        self.use_bpe = use_bpe
        self.bpe_length_threshold = bpe_length_threshold

        custom_pretokenizer = PreTokenizer.custom(CustomPreTokenizer())
        self.pre_tokenizer = Sequence([custom_pretokenizer, Whitespace()])
        self.tokenizer = BPETokenizer(50256, save_path=".cache/tokenizer.json").load()

    def __getstate__(self) -> object:
        data = self.__dict__.copy()
        del data["pre_tokenizer"]
        del data["tokenizer"]

        return data

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)

        custom_pretokenizer = PreTokenizer.custom(CustomPreTokenizer())
        self.pre_tokenizer = Sequence([custom_pretokenizer, Whitespace()])
        self.tokenizer = BPETokenizer(50256, save_path=".cache/tokenizer.json").load()

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def tokenize(self, text):
        text = self._split_mixed_tokens(text)
        words = self._split_symbols(text)
        words = self._split_camel_case(words)
        words = [
            split_word for word in words for split_word in self._split_snake_case(word)
        ]

        tokens = []
        for word in words:
            word = str(word).strip()
            if self.should_use_bpe(word):
                tokens.extend(self._split_bpe(word))
            else:
                tokens.append(
                    Term(
                        term=word,
                        original_term=word,
                        position=len(tokens),
                        start_char_offset=0,
                        end_char_offset=0,
                    )
                )

        return tokens

    def should_use_bpe(self, word):
        return self.use_bpe and len(word) > self.bpe_length_threshold

    def _split_bpe(self, text):
        identifiers = self.tokenizer.encode(text)
        tokens = []
        for identifier in identifiers:
            tokens.append(
                Term(
                    term=identifier,
                    original_term=text,
                    position=len(tokens),
                    start_char_offset=0,
                    end_char_offset=0,
                )
            )

        return tokens

    def _split_snake_case(self, text):
        return text.split("_")

    def _split_camel_case(self, text):
        return re.sub(self.camel_case_re, r"\1 \2", text).split(" ")

    def _split_mixed_tokens(self, text):
        identifier = re.sub(r"(\d+)", r" \1 ", text)
        return identifier

    def _split_symbols(self, text):
        tokens = re.findall(self.symbol_re, text)
        return " ".join(tokens)

    def _add_identifier(self, buffer, tokens, idx):
        encoded = self._encode(buffer)
        curr_start = idx - len(buffer)
        for token in encoded:
            tokens.append(
                Term(
                    term=token,
                    original_term=buffer,
                    position=len(tokens),
                    start_char_offset=curr_start,
                    end_char_offset=curr_start + len(token),
                )
            )
            curr_start += len(token)

    def _encode(self, text):
        return text.split(" ")


class ClassicTokenizer:
    """
    This tokenizer is based on the classic tokenizer in Elasticsearch
    https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-classic-tokenizer.html

    The following main points are applied:
        - Replaces hyphens with a whitespace (e.g. "High-performance" -> "High performance")
        - Recognises emails and URLs
        - Splits text based on regex defined word boundaries
        - Contractions are combined into a single token (e.g. "I've" -> "Ive")
        - Replaces double whitespace with a single whitespace

    We do not explictly remove punctuation as the regex pattern matches words and therefore punctuation is removed if it is not part of a word or we are not explicitly looking for it
    """

    def __init__(self):
        self.email_address_re = re.compile(
            r"(?:[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
        )
        self.url_re = re.compile(r"(?:(http|https)://[^\s]+)|(?:www\.[^\s]+)")

        self.token_re = re.compile(
            rf"""
                {self.url_re.pattern}
                {self.email_address_re.pattern}
                |(?:\b\w+'\w+\b)
                |(?:\b\w+\b)
            """,
            re.VERBOSE,
        )

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def tokenize(self, text, is_orderable=True):
        """
        Replaces all punctuation with an empty string
        Replaces all double whitespace with a single whitespace
        Splits the text into words
        Normalizes the words to NFKD form (decomposes the unicode characters)

        REFERENCE - Look at the following https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-classic-tokenizer.html
        """
        if not text or not text.strip() or len(text) == 0:
            return []

        tokens = self._tokenize(text, is_orderable=is_orderable)
        return tokens

    def _tokenize(self, text, is_orderable=True) -> list[Term]:
        tokens = []
        position = 0

        for match in self.token_re.finditer(text):
            token = match.group()
            start_offset = match.start()
            end_offset = match.end()

            if (
                "-" in token
                and not self._is_email_address(token)
                and not self._is_url(token)
            ):
                parts = token.split("-")
                for part in parts:
                    if part is None:
                        continue

                    tokens.append(
                        Term(
                            term=part,
                            original_term=part,
                            position=position if is_orderable else -1,
                            start_char_offset=start_offset if is_orderable else -1,
                            end_char_offset=end_offset if is_orderable else -1,
                        )
                    )
                    position += 1

                continue

            tokens.append(
                Term(
                    term=token,
                    original_term=token,
                    position=position if is_orderable else -1,
                    start_char_offset=start_offset if is_orderable else -1,
                    end_char_offset=end_offset if is_orderable else -1,
                )
            )
            position += 1

        return tokens

    def _is_email_address(self, text):
        return self.email_address_re.match(text)

    def _is_url(self, text):
        return self.url_re.match(text)


class LinkTokenizer:
    def __init__(self) -> None:
        self.tokenizer = ClassicTokenizer()

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def tokenize(self, text, href=None, alt_text=None):
        tokenized_text = self.tokenizer(text)
        href = self.tokenizer(href, is_orderable=False) if href else []
        alt_text = self.tokenizer(alt_text, is_orderable=False) if alt_text else []

        return tokenized_text + href + alt_text


class Tokenizer:
    def __init__(
        self,
        text_normalizer_operations=[],
        code_normalizer_operations=[],
        link_normalizer_operations=[],
    ):
        self.code_tokenizer = CodeTokenizer()
        self.text_tokenizer = ClassicTokenizer()
        self.link_tokenizer = LinkTokenizer()

        self.text_normalizer = Normalizer(text_normalizer_operations)
        self.code_normalizer = Normalizer(code_normalizer_operations)
        self.link_normalizer = Normalizer(link_normalizer_operations)

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def tokenize(self, text_block) -> TokenizedOutput:
        if text_block is None:
            return TokenizedOutput(tokenized_text=[], original_number_of_words=0)
        if text_block.text is None:
            return TokenizedOutput(tokenized_text=[], original_number_of_words=0)

        if isinstance(text_block, CodeBlock):
            tokenized_out = self.code_tokenizer(text_block.text)
            return TokenizedOutput(
                tokenized_text=self.code_normalizer(tokenized_out),
                original_number_of_words=len(tokenized_out),
            )

        elif isinstance(text_block, LinkBlock):
            tokenized_out = self.link_tokenizer(
                text_block.text, text_block.href, text_block.alt_text
            )
            return TokenizedOutput(
                tokenized_text=self.link_normalizer(tokenized_out),
                original_number_of_words=len(tokenized_out),
            )

        elif isinstance(text_block, TextBlock):
            tokenized_out = self.text_tokenizer(text_block.text)
            return TokenizedOutput(
                tokenized_text=self.text_normalizer(tokenized_out),
                original_number_of_words=len(tokenized_out),
            )
        else:
            raise ValueError(f"Unknown text block type: {type(text_block)}")
