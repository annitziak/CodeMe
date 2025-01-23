from dataclasses import dataclass, field
from enum import IntEnum


class TextSize(IntEnum):
    H1 = 1
    H2 = 2
    H3 = 3
    H4 = 4
    H5 = 5
    H6 = 6
    P = 7
    OL = 7
    UL = 7
    LI = 7


@dataclass
class Block:
    text: str
    block_id: int

    words: list[str] = field(default_factory=list)


@dataclass
class NormalTextBlock(Block):
    is_italic: bool = False
    is_bold: bool = False
    is_underline: bool = False

    text_size: int = TextSize.P


@dataclass
class LinkBlock(NormalTextBlock):
    href: str = ""
    alt_text: str = ""


@dataclass
class CodeBlock(Block):
    in_line: bool = False


@dataclass
class Term:
    term: str = ""
    original_term: str = ""  # before stemming/cleaning
    position: int = -1  # for phrase and word proximity
    start_char_offset: int = -1  # highlighting search snippets
    end_char_offset: int = -1  # highlighting search snippets
