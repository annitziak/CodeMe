import logging

from dataclasses import dataclass
from preprocessing import CodeBlock, NormalTextBlock as TextBlock, LinkBlock, Term
from preprocessing.normalizer import Normalizer
from preprocessing.sub_tokenizers.link_tokenizer import LinkTokenizer
from preprocessing.sub_tokenizers.main_tokenizer import (
    MainTokenizer,
    PunctuationHandler,
    DEFAULT_PUNCTUATION,
)

from preprocessing.normalizer import (
    LowerCaseNormalizer,
    StopWordNormalizer,
    StemmingNormalizer,
    UnicodeToAsciiNormalizer,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TokenizedOutput:
    tokenized_text: list[Term]
    original_number_of_words: int = 0


DEFAULT_TOKENIZER_KWARGS = (
    {
        "use_bpe": False,
        "link_behaviour": "remove",
        "mixed_token_behaviour": "split",
        "snake_case_behaviour": "split",
        "camel_case_behaviour": "split",
        "contractions_behaviour": "remove",
        "digit_behaviour": "remove",
        "punctuation_handlers": [
            PunctuationHandler(
                punctuation=DEFAULT_PUNCTUATION,
                punctuation_behaviour="remove-split",
            )
        ],
    },
)

DEFAULT_NORMALIZER_OPERATIONS = [
    LowerCaseNormalizer(),
    StopWordNormalizer(
        stop_words_file="preprocessing/stop_words/custom_tfidf_approach.txt"
    ),
    StemmingNormalizer(),
    UnicodeToAsciiNormalizer(),
]


class Tokenizer:
    def __init__(
        self,
        code_tokenizer_kwargs=DEFAULT_TOKENIZER_KWARGS,
        text_tokenizer_kwargs=DEFAULT_TOKENIZER_KWARGS,
        link_tokenizer_kwargs=DEFAULT_TOKENIZER_KWARGS,
        text_normalizer_operations=DEFAULT_NORMALIZER_OPERATIONS,
        code_normalizer_operations=DEFAULT_NORMALIZER_OPERATIONS,
        link_normalizer_operations=DEFAULT_NORMALIZER_OPERATIONS,
    ):
        self.code_tokenizer = MainTokenizer(code_tokenizer_kwargs)
        self.text_tokenizer = MainTokenizer(text_tokenizer_kwargs)
        self.link_tokenizer = LinkTokenizer(link_tokenizer_kwargs)

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
            print(tokenized_out)
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
