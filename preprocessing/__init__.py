from dataclasses import dataclass, field
from enum import IntEnum


class TextSize(IntEnum):
    UNK = -1
    H1 = 1
    H2 = 2
    H3 = 3
    H4 = 4
    H5 = 5
    H6 = 6
    P = 7


@dataclass
class Block:
    text: str
    block_id: int

    words: list[str] = field(default_factory=list)

    def update(self, block, *args, **kwargs):
        if block is None:
            return self

        for key, value in block.__dict__.items():
            # Only set the value if it is a boolean and is True
            if isinstance(value, bool):
                if value:
                    setattr(self, key, value)

            # Only set the value if it is an instance of TextSize and the current value is UNK
            if hasattr(self, key):
                if (
                    isinstance(value, TextSize)
                    and value != TextSize.UNK
                    and getattr(self, key) == TextSize.UNK
                ):
                    setattr(self, key, value)

        return self

    def __iter__(self):
        for word in self.words:
            yield word


@dataclass
class NormalTextBlock(Block):
    """
    HTML tag elements are listed here: https://meta.stackexchange.com/questions/1777/what-html-tags-are-allowed-on-stack-exchange-sites
    """

    is_italic: bool = False
    is_bold: bool = False
    is_underline: bool = False
    is_superscript: bool = False
    is_blockquote: bool = False
    is_strike_through: bool = False
    is_deleted: bool = False
    is_desciption_list: bool = False
    is_emphasis: bool = False
    is_idiomatic: bool = False
    is_list: bool = False

    text_size: int = TextSize.UNK

    def update(self, *args, **kwargs):
        super().update(self, *args, **kwargs)


@dataclass
class LinkBlock(NormalTextBlock):
    href: str = ""
    alt_text: str = ""

    def update(self, *args, **kwargs):
        super().update(self, *args, **kwargs)


@dataclass
class CodeBlock(Block):
    in_line: bool = False

    def update(self, *args, **kwargs):
        super().update(self, *args, **kwargs)


@dataclass
class Term:
    term: str = ""
    original_term: str = ""  # before stemming/cleaning
    position: int = -1  # for phrase and word proximity
    start_char_offset: int = -1  # highlighting search snippets
    end_char_offset: int = -1  # highlighting search snippets
