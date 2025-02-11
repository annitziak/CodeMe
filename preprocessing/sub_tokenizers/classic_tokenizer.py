import re

from preprocessing import Term


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
