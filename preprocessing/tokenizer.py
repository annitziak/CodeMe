import re
import logging
import unicodedata

from itertools import zip_longest
from preprocessing import CodeBlock, NormalTextBlock as TextBlock, LinkBlock, Term

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CodeTokenizer:
    def tokenize(self, text):
        return text


class WhitespaceTokenizer:
    def __init__(self):
        self.split_hypen_re = re.compile(r"[\w]+-[\w]+")

        self.email_address_re = re.compile(
            r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
        )
        self.url_re1 = re.compile(r"(http|https)://[^\s]+")
        self.url_re2 = re.compile(r"www\.[^\s]+")
        self.dot_in_word_not_followed_by_whitespace = re.compile(r"\w+\.\w+")

        self.punctuation_re = re.compile(r"[^\w\s]")
        self.double_space_re = re.compile(r"\s\s+")

    def tokenize(self, text):
        """
        Replaces all punctuation with an empty string
        Replaces all double whitespace with a single whitespace
        Splits the text into words
        Normalizes the words to NFKD form (decomposes the unicode characters)

        REFERENCE - Look at the following https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-classic-tokenizer.html
        """
        if not text:
            return []

        original_term_offsets = self._get_offsets(text)

        text, email_addresses = self._remove_single_tokens(
            self.email_address_re, text, placeholder="|| EMAIL ||"
        )
        text, urls1 = self._remove_single_tokens(
            self.url_re1, text, placeholder="|| URL1 ||"
        )
        text, urls2 = self._remove_single_tokens(
            self.url_re2, text, placeholder="|| URL2 ||"
        )
        text, dot_words = self._remove_single_tokens(
            self.dot_in_word_not_followed_by_whitespace,
            text,
            placeholder="|| DOT_WORD ||",
        )

        cleaned_text = self.punctuation_re.sub("", text)

        cleaned_text = self._add_single_tokens(
            email_addresses, cleaned_text, "|| EMAIL ||"
        )
        cleaned_text = self._add_single_tokens(urls1, cleaned_text, "|| URL1 ||")
        cleaned_text = self._add_single_tokens(urls2, cleaned_text, "|| URL2 ||")
        cleaned_text = self._add_single_tokens(
            dot_words, cleaned_text, "|| DOT_WORD ||"
        )

        cleaned_text = self.double_space_re.sub(" ", cleaned_text)
        words = re.split(r"\s+", cleaned_text.strip().lower())
        words = [unicodedata.normalize("NFKD", word) for word in words]
        updated_offsets = self._update_offsets(original_term_offsets, words)

        return updated_offsets

    def _update_positions(self, original_offsets: list[Term], words: list[str]):
        """
        Given the original offsets of the words in the text, we can update
        the positions to reflect the new positions of the words after tokenization
        Also, we update the `term` property of the Term object to reflect the tokenized output

        For example, if the original text was "high-quality" then it would be tokenized
        as ["high", "quality"] and the positions would be [0, 1] respectively
        """
        current_offset = 0
        updated_offsets= []

        assert len(original_offsets) <= len(words), "The number of words in the text should be greater than or equal to the number of original offsets. New words should only be created"


        offset_idx = 0
        for word_idx, word in enumerate(words):





        return updated_offsets

    def _get_offsets(self, text):
        """
        Do a simple split on the text to obtain the start and end offsets of each word
        This is used to keep track of the original offsets of the words such that we can
        obtain their original position in the text
        """
        words = text.split()
        current_offset = 0
        offsets = []

        for idx, word in enumerate(words):
            start = text.find(word, current_offset)
            end = start + len(word)
            current_offset = end

            offsets.append(
                Term(
                    term=word,
                    original_term=word,
                    position=idx,
                    start_char_offset=start,
                    end_char_offset=end,
                )
            )

        return offsets

    def _remove_single_tokens(self, regex, text, placeholder="|| SPECIAL ||"):
        tokens = re.findall(regex, text)
        replaced_text = re.sub(regex, placeholder, text)

        return replaced_text, tokens

    def _add_single_tokens(self, tokens, text, placeholder="|| SPECIAL ||"):
        for token in tokens:
            text = text.replace(placeholder, token, 1)

        return text


class LinkTokenizer:
    def tokenize(self, text, href=None, alt_text=None):
        return text


class Tokenizer:
    def __init__(self):
        self.code_tokenizer = CodeTokenizer()
        self.text_tokenizer = WhitespaceTokenizer()
        self.link_tokenizer = LinkTokenizer()

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def tokenize(self, text_block):
        if isinstance(text_block, CodeBlock):
            return self.code_tokenizer.tokenize(text_block.text)
        elif isinstance(text_block, TextBlock):
            return self.text_tokenizer.tokenize(text_block.text)
        elif isinstance(text_block, LinkBlock):
            return self.link_tokenizer.tokenize(text_block.text)
