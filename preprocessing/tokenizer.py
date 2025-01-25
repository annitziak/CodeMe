import re
import logging

from preprocessing import CodeBlock, NormalTextBlock as TextBlock, LinkBlock, Term
from preprocessing.normalizer import Normalizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CodeTokenizer:
    def __init__(self):
        """
        Naive tokenizer for code snippets

        Based on keywords and operators found
        """
        self.literal_re = re.compile(
            r'(\d+|".*?|\'.*?\'|True|False|true|false|null|None|NaN|Inf)'
        )
        self.symbols = set("+-*/=()[]{}<>:;,.!@#$%^&|~`")

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def tokenize(self, text):
        tokens = []
        buffer = ""
        for idx, char in enumerate(text):
            if not char.isspace() and char not in self.symbols:
                buffer += char
                continue

            if len(buffer) > 0:
                if not self._add_literal(buffer, tokens, idx):
                    self._add_identifier(buffer, tokens, idx)

                buffer = ""

            if char in self.symbols:
                tokens.append(
                    Term(
                        term=char,
                        original_term=char,
                        position=len(tokens),
                        start_char_offset=idx,
                        end_char_offset=idx + 1,
                    )
                )

        if len(buffer) > 0:
            if not self._add_literal(buffer, tokens, len(text)):
                self._add_identifier(buffer, tokens, len(text))

        return tokens

    def _add_literal(self, buffer, tokens, idx):
        if self.literal_re.match(buffer):
            tokens.append(
                Term(
                    term=buffer,
                    original_term=buffer,
                    position=len(tokens),
                    start_char_offset=idx - len(buffer),
                    end_char_offset=idx,
                )
            )
            return True

        return False

    def _add_identifier(self, buffer, tokens, idx):
        encoded = self._encode(buffer)
        curr_end = idx
        for token in encoded[::-1]:
            tokens.append(
                Term(
                    term=token,
                    original_term=buffer,
                    position=len(tokens),
                    start_char_offset=curr_end - len(token),
                    end_char_offset=curr_end,
                )
            )
            curr_end -= len(token)

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

    def tokenize(self, text_block):
        if isinstance(text_block, CodeBlock):
            return self.code_normalizer(self.code_tokenizer(text_block.text))
        elif isinstance(text_block, LinkBlock):
            return self.link_normalizer(
                self.link_tokenizer(
                    text_block.text, text_block.href, text_block.alt_text
                )
            )
        elif isinstance(text_block, TextBlock):
            return self.text_normalizer(self.text_tokenizer(text_block.text))
