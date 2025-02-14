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
        if self.punctuation_behaviour == "keep-split":
            self.re = re.compile(f"([{pattern}])")

    def __call__(self, positions: list[Term]):
        if self.punctuation_behaviour == "remove-split":
            return self._handle_remove_split(positions, self.re)
        elif self.punctuation_behaviour == "keep":
            return positions
        elif self.punctuation_behaviour == "remove":
            return self._handle_remove(positions, self.re)
        elif self.punctuation_behaviour == "keep-split":
            # record the words that should also be split
            return self._handle_keep_split(positions, self.re)
        else:
            raise ValueError(
                f"Unknown punctuation behaviour: {self.punctuation_behaviour}"
            )

    def _handle_remove_split(self, positions: list[Term], regex: str | re.Pattern):
        new_positions = []
        for pos in positions:
            parts = regex.split(pos.term)
            current_pos = pos.start_position
            current_offset = pos.start_char_offset

            for part in parts:
                if part:
                    new_positions.append(
                        Term(
                            term=part,
                            original_term=pos.original_term,
                            start_position=current_pos,
                            end_position=current_pos,
                            start_char_offset=current_offset,
                            end_char_offset=current_offset + len(part),
                        )
                    )
                    current_pos += 1
                    current_offset += len(part) + 1

        return new_positions

    def _handle_keep_split(self, positions: list[Term], regex: str | re.Pattern):
        new_positions = []
        for pos in positions:
            new_positions.append(
                Term(
                    term=pos.term,
                    original_term=pos.original_term,
                    start_position=pos.start_position,
                    end_position=pos.end_position,
                    start_char_offset=pos.start_char_offset,
                    end_char_offset=pos.end_char_offset,
                )
            )
            parts = regex.split(pos.term)
            current_pos = pos.start_position
            current_offset = pos.start_char_offset

            for part in parts:
                if part:
                    new_positions.append(
                        Term(
                            term=part,
                            original_term=pos.original_term,
                            start_position=current_pos,
                            end_position=current_pos,
                            start_char_offset=current_offset,
                            end_char_offset=current_offset + len(part),
                        )
                    )
                    current_pos += 1
                    current_offset += len(part) + 1
        return new_positions

    def _handle_remove(self, positions: list[Term], regex: str | re.Pattern):
        new_positions = []
        for pos in positions:
            parts = regex.split(pos.term)
            current_pos = pos.start_position
            current_offset = pos.start_char_offset

            new_term = "".join(parts)
            if new_term:
                new_positions.append(
                    Term(
                        term=new_term,
                        original_term=pos.original_term,
                        start_position=current_pos,
                        end_position=current_pos,
                        start_char_offset=current_offset,
                        end_char_offset=current_offset + len(new_term),
                    )
                )
                current_pos += 1
                current_offset += len(new_term)

        return new_positions


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
                `split` - split the tokens into words and remove numbers (e.g. "abc 123 def")
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

        self.whitespace_re = re.compile(r"\s+")

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
        words = self._initial_split(text)
        words = self._handle_links(words)
        words = self._handle_contractions(words)
        words = self._handle_mixed_tokens(words)
        words = self._handle_digits(words)
        words = self._split_symbols(words)
        words = self._handle_camel_case(words)
        words = self._split_snake_case(words)

        return words

    def _initial_split(self, text: str):
        last_end = 0
        positions = []
        for match in self.whitespace_re.finditer(text):
            start, end = match.span()
            if start > last_end:
                token_text = text[last_end:start]
                positions.append(
                    Term(
                        term=token_text,
                        original_term=token_text,
                        start_position=len(positions),
                        end_position=len(positions),
                        start_char_offset=last_end,
                        end_char_offset=start,
                    )
                )
            last_end = end

        if last_end < len(text):
            token_text = text[last_end:]
            positions.append(
                Term(
                    term=token_text,
                    original_term=token_text,
                    start_position=len(positions),
                    end_position=len(positions),
                    start_char_offset=last_end,
                    end_char_offset=len(text),
                )
            )

        return positions

    def _handle_digits(self, positions: list[Term]):
        if self.digit_behaviour == "remove":
            new_positions = []
            for pos in positions:
                new_term = re.sub(r"\d", "", pos.term)
                if new_term:
                    new_positions.append(
                        Term(
                            term=new_term,
                            original_term=pos.original_term,
                            start_position=pos.start_position,
                            end_position=pos.end_position,
                            start_char_offset=pos.start_char_offset,
                            end_char_offset=pos.end_char_offset,
                        )
                    )
            return new_positions
        elif self.digit_behaviour == "keep":
            return positions

    def _handle_contractions(self, positions: list[Term]):
        if self.contractions_behaviour == "split":
            return self._handle_general_split(positions, self.contraction_re)
        elif self.contractions_behaviour == "keep":
            return positions
        elif self.contractions_behaviour == "remove":
            new_positions = []
            for pos in positions:
                new_positions.append(
                    Term(
                        term=self.contraction_re.sub(r"\1\2", pos.term),
                        original_term=pos.original_term,
                        start_position=pos.start_position,
                        end_position=pos.end_position,
                        start_char_offset=pos.start_char_offset,
                        end_char_offset=pos.end_char_offset,
                    )
                )
            return new_positions
        elif self.contractions_behaviour == "keep-split":
            return self._handle_general_keep_split(positions, self.contraction_re)
        else:
            raise ValueError(
                f"Unknown contractions behaviour: {self.contractions_behaviour}"
            )

    def _handle_links(self, positions: list[Term]):
        new_positions = []
        if self.link_behaviour == "remove":
            for pos in positions:
                if self.url_re.match(pos.term) or self.email_address_re.match(pos.term):
                    continue
                else:
                    new_positions.append(pos)
        elif self.link_behaviour == "keep":
            return positions
        else:
            raise ValueError(f"Unknown link behaviour: {self.link_behaviour}")

        return new_positions

    def _split_snake_case(self, positions: list[Term]):
        if self.snake_case_behaviour == "split":
            return self._handle_general_split(positions, r"_")
        elif self.snake_case_behaviour == "keep":
            return positions
        elif self.snake_case_behaviour == "keep-split":
            # record the words that should also be split
            return self._handle_general_keep_split(positions, r"_")
        else:
            raise ValueError(
                f"Unknown snake case behaviour: {self.snake_case_behaviour}"
            )

    def _handle_general_split(self, positions: list[Term], regex: str | re.Pattern):
        new_positions = []
        for pos in positions:
            parts = regex.split(pos.term)
            current_pos = pos.start_position
            current_offset = pos.start_char_offset

            for part in parts:
                if part:
                    new_positions.append(
                        Term(
                            term=part,
                            original_term=pos.original_term,
                            start_position=current_pos,
                            end_position=current_pos,
                            start_char_offset=current_offset,
                            end_char_offset=current_offset + len(part),
                        )
                    )
                    current_pos += 1
                    current_offset += len(part) + 1

        return new_positions

    def _handle_general_keep_split(
        self, positions: list[Term], regex: str | re.Pattern
    ):
        new_positions = []
        for pos in positions:
            new_positions.append(
                Term(
                    term=pos.term,
                    original_term=pos.original_term,
                    start_position=pos.start_position,
                    end_position=pos.end_position,
                    start_char_offset=pos.start_char_offset,
                    end_char_offset=pos.end_char_offset,
                )
            )
            parts = regex.split(pos.term)
            current_pos = pos.start_position
            current_offset = pos.start_char_offset

            for part in parts:
                if part:
                    new_positions.append(
                        Term(
                            term=part,
                            original_term=pos.original_term,
                            start_position=current_pos,
                            end_position=current_pos,
                            start_char_offset=current_offset,
                            end_char_offset=current_offset + len(part),
                        )
                    )
                    current_pos += 1
                    current_offset += len(part) + 1

        return new_positions

    def _handle_camel_case(self, positions: list[Term]):
        if self.camel_case_behaviour == "split":
            return self._handle_general_split(positions, self.camel_case_re)
        elif self.camel_case_behaviour == "keep":
            return positions
        elif self.camel_case_behaviour == "keep-split":
            # record the words that should also be split
            return self._handle_general_keep_split(positions, self.camel_case_re)
        else:
            raise ValueError(
                f"Unknown camel case behaviour: {self.camel_case_behaviour}"
            )

    def _handle_mixed_tokens(self, positions: list[Term]):
        if self.mixed_token_behaviour == "split":
            return self._handle_general_split(positions, r"(\d+)")
        elif self.mixed_token_behaviour == "keep":
            return positions
        elif self.mixed_token_behaviour == "keep-split":
            # record the words that should also be split
            return self._handle_general_keep_split(positions, r"(\d+)")
        else:
            raise ValueError(
                f"Unknown mixed token behaviour: {self.mixed_token_behaviour}"
            )

    def _split_symbols(self, positions: list[Term]):
        for handler in self.punctuation_handlers:
            positions = handler(positions)

        return positions

    def _encode(self, text):
        return text.split(" ")
