import re

from tokenizers.pre_tokenizers import PreTokenizer, Whitespace, Sequence

from preprocessing import Term
from preprocessing.sub_tokenizers.bpe_code_tokenizer import (
    CustomPreTokenizer,
    BPETokenizer,
)

DEFAULT_PUNCTUATION = [
    "[",
    "]",
    "{",
    "}",
    "<",
    ">",
    "(",
    ")",
    ".",
    ",",
    ";",
    ":",
    "=",
    "+",
    "-",
    "*",
    "/",
    "!",
    "@",
    "#",
    "$",
    "%",
    "^",
    "&",
    "~",
    "`",
    "#",
    "?",
    '"',
    "'",
    "\\",
]


class PunctuationHandler:
    def __init__(self, punctuation=[], punctuation_behaviour="remove-split"):
        self.punctuation_behaviour = punctuation_behaviour

        self.punctuation = punctuation
        if isinstance(self.punctuation, str):
            self.punctuation = self.punctuation.split()

        self.punctuation = [re.escape(p) for p in self.punctuation]
        pattern = "".join(self.punctuation)

        self.re = re.compile(f"[{pattern}]")

    def __call__(self, text):
        if self.punctuation_behaviour == "remove-split":
            return self.re.sub(r"  ", text)
        elif self.punctuation_behaviour == "keep":
            return text
        elif self.punctuation_behaviour == "remove":
            return self.re.sub("", text)
        elif self.punctuation_behaviour == "keep-split":
            # record the words that should also be split
            raise NotImplementedError(
                "Punctuation behaviour not implemented. To keep the original and tokenized punctuation"
            )
        else:
            raise ValueError(
                f"Unknown punctuation behaviour: {self.punctuation_behaviour}"
            )


class MainTokenizer:
    def __init__(
        self,
        use_bpe=False,
        bpe_length_threshold=15,
        link_behaviour="remove",
        mixed_token_behaviour="split",
        snake_case_behaviour="split",
        camel_case_behaviour="split",
        contractions_behaviour="remove",
        digit_behaviour="remove",
        punctuation_handlers=[
            PunctuationHandler(
                punctuation=DEFAULT_PUNCTUATION,
                punctuation_behaviour="remove-split",
            )
        ],
    ):
        """
        Args:
            mixed_token_behaviour (str): How to handle mixed tokens (e.g. "abc123def")
                `split` - split the tokens into words (e.g. "abc 123 def")
                `keep` - keep the tokens as is (e.g. "abc123def")
                `keep-split` - keep the tokens as is but also split them (e.g. "abc 123 def" & "abc123def")
            snake_case_behaviour (str): How to handle snake case tokens (e.g. "snake_case")
                `split`, `keep`, `keep-split`
            camel_case_behaviour (str): How to handle camel case tokens (e.g. "camelCase")
                `split`, `keep`, `keep-split`
            contractions_behaviour (str): How to handle contractions (e.g. "I've")
                `split` - split the contractions into words (e.g. "I've" -> "I ve")
                `keep` - keep the contractions as is (e.g. "I've")
                `remove` - remove the apostrophe (e.g. "I've" -> "Ive")
                `keep-split` - keep the contractions as is but also split them (e.g. "I've" -> "Ive" & "I've")
            digit_behaviour (str): How to handle digits. Be careful in relation to the `mixed_token_behaviour`
                `remove` - remove the digits (e.g. "abc123def" -> "abcdef")
                `keep` - keep the digits (e.g. "abc123def")
            punctuation_behaviour (str): How to handle punctuation. Specifies list of handlers to use to handle punctuation. Each has its own punctuation and the behaviour to apply to it


        """
        assert link_behaviour in ["remove", "keep", "keep-split", "split"]
        assert contractions_behaviour in ["remove", "keep", "split"]

        # CGPath -> CG Path
        # cgPath -> cg Path
        # cgpAth -> cgp Ath
        # CGPath2D -> CG Path 2D
        self.camel_case_re = re.compile(
            r"(?<!^)(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z]))"
        )
        self.punctuation_handlers = punctuation_handlers

        self.email_address_re = re.compile(
            r"(?:[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
        )
        self.url_re = re.compile(r"(?:(http|https)://[^\s]+)|(?:www\.[^\s]+)")
        self.contraction_re = re.compile(r"(\b\w+)'(\w+\b)")

        self.use_bpe = use_bpe
        self.bpe_length_threshold = bpe_length_threshold

        self.link_behaviour = link_behaviour
        self.mixed_token_behaviour = mixed_token_behaviour
        self.snake_case_behaviour = snake_case_behaviour
        self.camel_case_behaviour = camel_case_behaviour
        self.contractions_behaviour = contractions_behaviour
        self.digit_behaviour = digit_behaviour

        custom_pretokenizer = PreTokenizer.custom(CustomPreTokenizer())
        self.pre_tokenizer = Sequence([custom_pretokenizer, Whitespace()])
        self.tokenizer = BPETokenizer(50256, save_path=".cache/tokenizer.json").load()

        self.keep_split_words = {}

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

    def tokenize(self, text, is_orderable=True):
        text = self._handle_links(text)
        text = self._handle_contractions(text)
        text = self._split_mixed_tokens(text)
        text = self._handle_digits(text)
        words = self._split_symbols(text)
        words = self._split_camel_case(words)
        words = [
            split_word for word in words for split_word in self._split_snake_case(word)
        ]
        words = [re.sub(r"\s+|\n", " ", word) for word in words]

        tokens = []
        curr_position = 0
        for word in words:
            word = str(word).strip()
            if len(word) == 0:
                continue

            if self.should_use_bpe(word):
                tokens.extend(self._split_bpe(word))
            else:
                tokens.append(
                    Term(
                        term=word,
                        original_term=word,
                        position=curr_position if is_orderable else -1,
                        start_char_offset=0,
                        end_char_offset=0,
                    )
                )

                if is_orderable:
                    curr_position += 1

        return tokens

    def _handle_digits(self, text):
        if self.digit_behaviour == "remove":
            return re.sub(r"(\d+)", r" ", text)
        elif self.digit_behaviour == "keep":
            return text

    def _handle_contractions(self, text):
        if self.contractions_behaviour == "split":
            return self.contraction_re.sub(r"\1 \2", text)
        elif self.contractions_behaviour == "keep":
            return text
        elif self.contractions_behaviour == "remove":
            return self.contraction_re.sub(r"\1\2", text)
        elif self.contractions_behaviour == "keep-split":
            # record the words that should also be split
            raise NotImplementedError(
                "Contractions behaviour not implemented. To keep the original and tokenized contractions"
            )
        else:
            raise ValueError(
                f"Unknown contractions behaviour: {self.contractions_behaviour}"
            )

    def _handle_links(self, text):
        if self.link_behaviour == "remove":
            subbed = re.sub(self.url_re, "", text)
            return re.sub(self.email_address_re, "", subbed)
        elif self.link_behaviour == "keep":
            return text

        res = [self.url_re, self.email_address_re]
        for regex_exp in res:
            for match in regex_exp.finditer(text):
                word = match.group()
                if self.link_behaviour == "keep-split":
                    raise NotImplementedError(
                        "Link behaviour not implemented. To keep the original and tokenized link"
                    )
                elif self.link_behaviour == "split":
                    raise NotImplementedError(
                        "Link behaviour not implemented. To only keep the tokenized link"
                    )
                else:
                    raise ValueError(f"Unknown link behaviour: {self.link_behaviour}")

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
        if self.snake_case_behaviour == "split":
            return text.split("_")
        elif self.snake_case_behaviour == "keep":
            return text
        elif self.snake_case_behaviour == "keep-split":
            # record the words that should also be split
            raise NotImplementedError(
                "Snake case behaviour not implemented. To keep the original and tokenized snake case"
            )
        else:
            raise ValueError(
                f"Unknown snake case behaviour: {self.snake_case_behaviour}"
            )

    def _split_camel_case(self, text):
        if len(text) == 0:
            return []

        if self.camel_case_behaviour == "split":

            def process_parts(parts):
                result = []
                for part in parts:
                    if part.isupper() and len(part) > 2:
                        # Find where uppercase run ends
                        i = 0
                        while i < len(part) and part[i].isupper():
                            i += 1
                        if i < len(part):  # Found lowercase
                            result.extend([part[:i], part[i:]])
                        else:
                            result.append(part)
                    else:
                        result.append(part)
                return result

            # Split and process
            parts = " ".join(re.split(self.camel_case_re, text)).split()
            return parts

        elif self.camel_case_behaviour == "keep":
            return text.split(" ")
        elif self.camel_case_behaviour == "keep-split":
            # record the words that should also be split
            raise NotImplementedError(
                "Camel case behaviour not implemented. To keep the original and tokenized camel case"
            )
        else:
            raise ValueError(
                f"Unknown camel case behaviour: {self.camel_case_behaviour}"
            )

    def _split_mixed_tokens(self, text):
        if self.mixed_token_behaviour == "split":
            return re.sub(r"(\d+)", r" \1 ", text)
        elif self.mixed_token_behaviour == "keep":
            return text
        elif self.mixed_token_behaviour == "keep-split":
            # record the words that should also be split
            raise NotImplementedError(
                "Mixed token behaviour not implemented. To keep the original and tokenized mixed token"
            )
        else:
            raise ValueError(
                f"Unknown mixed token behaviour: {self.mixed_token_behaviour}"
            )

    def _split_symbols(self, text):
        for handler in self.punctuation_handlers:
            text = handler(text)

        return text

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
