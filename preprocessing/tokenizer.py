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
        token_patterns = {
            "identifier": re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"),
            "number": re.compile(r"\b\d+(\.\d+)?\b"),
            "string": re.compile(r"\".*?\""),
            "comment": re.compile(r"#.*"),
            "operator": re.compile(r"[-+*/%&|^~<>!]=?"),
            "punctuation": re.compile(r"[.,;:()\[\]{}\\]"),
            "whitespace": re.compile(r"\s+"),
        }

        self.token_re = re.compile(
            "|".join(
                f"(?P<{name}>{pattern.pattern})"
                for name, pattern in token_patterns.items()
            )
        )

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def tokenize(self, text):
        tokens = []
        for match in self.token_re.finditer(text):
            for _, pattern in match.groupdict().items():
                if pattern is not None:
                    if pattern.strip() == "":
                        continue

                    tokens.append(
                        Term(
                            term=pattern,
                            original_term=pattern,
                            position=match.start(),
                            start_char_offset=match.start(),
                            end_char_offset=match.end(),
                        )
                    )
                    break

        return tokens


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

    def tokenize(self, text):
        """
        Replaces all punctuation with an empty string
        Replaces all double whitespace with a single whitespace
        Splits the text into words
        Normalizes the words to NFKD form (decomposes the unicode characters)

        REFERENCE - Look at the following https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-classic-tokenizer.html
        """
        if not text or not text.strip() or len(text) == 0:
            return []

        tokens = self._tokenize(text)
        return tokens

    def _tokenize(self, text) -> list[Term]:
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
                            position=position,
                            start_char_offset=start_offset,
                            end_char_offset=end_offset,
                        )
                    )
                    position += 1

                continue

            tokens.append(
                Term(
                    term=token,
                    original_term=token,
                    position=position,
                    start_char_offset=start_offset,
                    end_char_offset=end_offset,
                )
            )
            position += 1

        return tokens

    def _is_email_address(self, text):
        return self.email_address_re.match(text)

    def _is_url(self, text):
        return self.url_re.match(text)


class LinkTokenizer:
    def tokenize(self, text, href=None, alt_text=None):
        return text


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
        elif isinstance(text_block, TextBlock):
            return self.text_normalizer(self.text_tokenizer(text_block.text))
        elif isinstance(text_block, LinkBlock):
            return self.link_normalizer(self.link_tokenizer(text_block.text))
