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


DEFAULT_TOKENIZER_KWARGS = {
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
        ),
    ],
}


DEFAULT_NORMALIZER_OPERATIONS = [
    LowerCaseNormalizer(),
    StopWordNormalizer(
        stop_words_file="preprocessing/stop_words/custom_tfidf_approach.txt"
    ),
    StemmingNormalizer(),
    UnicodeToAsciiNormalizer(),
]

DEFAULT_PRE_TEXT_NORMALIZER_OPERATIONS = [
    UnicodeToAsciiNormalizer(),
]

"""
            PunctuationHandler(
                punctuation=r"(\S+)\.(\S+)",
                punctuation_behaviour="remove-split",
            ),
            PunctuationHandler(
                punctuation=PUNCTUATION_DOT,
                punctuation_behaviour="remove-split",
            ),
"""


def BuildTokenizer(tokenizer_type: str):
    if tokenizer_type == "keep-split-variables":
        tokenizer_kwargs = DEFAULT_TOKENIZER_KWARGS
        tokenizer_kwargs["punctuation_handlers"] = [
            PunctuationHandler(
                punctuation=DEFAULT_PUNCTUATION,
                punctuation_behaviour="remove-split",
            )
        ]
        tokenizer_kwargs["camel_case_behaviour"] = "keep-split"
        tokenizer_kwargs["snake_case_behaviour"] = "keep-split"
    elif tokenizer_type == "split-variables":
        tokenizer_kwargs = DEFAULT_TOKENIZER_KWARGS
        tokenizer_kwargs["punctuation_handlers"] = [
            PunctuationHandler(
                punctuation=DEFAULT_PUNCTUATION,
                punctuation_behaviour="remove-split",
            ),
        ]
        tokenizer_kwargs["camel_case_behaviour"] = "split"
        tokenizer_kwargs["snake_case_behaviour"] = "split"
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    return tokenizer_kwargs


def BuildNormalizer(normalizer_type: str):
    normalizer_operations = DEFAULT_NORMALIZER_OPERATIONS
    if normalizer_type == "no-stopwords":
        normalizer_operations = [
            LowerCaseNormalizer(),
            StemmingNormalizer(),
            UnicodeToAsciiNormalizer(),
        ]

    return normalizer_operations


class Tokenizer:
    def __init__(
        self,
        pre_text_normalizer_operations=DEFAULT_PRE_TEXT_NORMALIZER_OPERATIONS,
        pre_code_normalizer_operations=DEFAULT_PRE_TEXT_NORMALIZER_OPERATIONS,
        pre_link_normalizer_operations=DEFAULT_PRE_TEXT_NORMALIZER_OPERATIONS,
        code_tokenizer_kwargs=DEFAULT_TOKENIZER_KWARGS,
        text_tokenizer_kwargs=DEFAULT_TOKENIZER_KWARGS,
        link_tokenizer_kwargs=DEFAULT_TOKENIZER_KWARGS,
        post_text_normalizer_operations=DEFAULT_NORMALIZER_OPERATIONS,
        post_code_normalizer_operations=DEFAULT_NORMALIZER_OPERATIONS,
        post_link_normalizer_operations=DEFAULT_NORMALIZER_OPERATIONS,
    ):
        self.code_tokenizer = MainTokenizer(**code_tokenizer_kwargs)
        self.text_tokenizer = MainTokenizer(**text_tokenizer_kwargs)
        self.link_tokenizer = LinkTokenizer(**link_tokenizer_kwargs)

        self.pre_text_normalizer = Normalizer(pre_text_normalizer_operations)
        self.pre_code_normalizer = Normalizer(pre_code_normalizer_operations)
        self.pre_link_normalizer = Normalizer(pre_link_normalizer_operations)

        self.post_text_normalizer = Normalizer(post_text_normalizer_operations)
        self.post_code_normalizer = Normalizer(post_code_normalizer_operations)
        self.post_link_normalizer = Normalizer(post_link_normalizer_operations)

    def __repr__(self):
        return f"Tokenizer(code_tokenizer={self.code_tokenizer}, text_tokenizer={self.text_tokenizer}, link_tokenizer={self.link_tokenizer}, post_text_normalizer={self.post_text_normalizer}, post_code_normalizer={self.post_code_normalizer}, post_link_normalizer={self.post_link_normalizer})"

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def tokenize(self, text_block) -> TokenizedOutput:
        if text_block is None:
            return TokenizedOutput(tokenized_text=[], original_number_of_words=0)
        if text_block.text is None:
            return TokenizedOutput(tokenized_text=[], original_number_of_words=0)

        if isinstance(text_block, CodeBlock):
            text = self.pre_code_normalizer(text_block.text)
            tokenized_out = self.code_tokenizer(text)
            return TokenizedOutput(
                tokenized_text=self.post_code_normalizer(tokenized_out),
                original_number_of_words=len(tokenized_out),
            )

        elif isinstance(text_block, LinkBlock):
            text = self.pre_link_normalizer(text_block.text)
            tokenized_out = self.link_tokenizer(
                text, text_block.href, text_block.alt_text
            )
            return TokenizedOutput(
                tokenized_text=self.post_link_normalizer(tokenized_out),
                original_number_of_words=len(tokenized_out),
            )

        elif isinstance(text_block, TextBlock):
            text = self.pre_text_normalizer(text_block.text)
            tokenized_out = self.text_tokenizer(text)
            return TokenizedOutput(
                tokenized_text=self.post_text_normalizer(tokenized_out),
                original_number_of_words=len(tokenized_out),
            )
        else:
            raise ValueError(f"Unknown text block type: {type(text_block)}")
