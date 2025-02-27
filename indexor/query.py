import re
import logging

from dataclasses import dataclass

from preprocessing import Block
from preprocessing.preprocessor import Preprocessor

DEFAULT_PREPROCESSOR = Preprocessor(parser_kwargs={"parser_type": "raw"})

logger = logging.getLogger(__name__)


class Query:
    def __init__(self, query, preprocessor=DEFAULT_PREPROCESSOR):
        self.query = query
        self.preprocessor = preprocessor

    def parse(self):
        return self.query

    def pprint(self, level=0):
        print(self.ppformat(level=level))

    def ppformat(self, level=0):
        return " " * level + self.query + "\n"

    def __str__(self):
        return self.ppformat()


class TermQuery(Query):
    def __init__(self, query, preprocessor=DEFAULT_PREPROCESSOR):
        super().__init__(query, preprocessor)

        parsed_query = self.parse()
        parsed_query = parsed_query[0] if parsed_query else None
        self.parsed_query = (
            parsed_query.words[0].term if parsed_query and parsed_query.words else ""
        )

        print(f"TERM QUERY: {self.query} -> {parsed_query} -> {self.parsed_query}")

    def parse(self):
        return self.preprocessor(self.query)

    def ppformat(self, level=0):
        return " " * level + self.parsed_query + "\n"


class FreeTextQuery(Query):
    def __init__(self, query, preprocessor=DEFAULT_PREPROCESSOR):
        super().__init__(query, preprocessor)

        self.parsed_query = self.parse()

    def parse(self):
        def _parse():
            if isinstance(self.query, str):
                return self.preprocessor(self.query)
            elif isinstance(self.query, list):
                if all(isinstance(token, Block) for token in self.query):
                    return self.query
                elif all(isinstance(token, str) for token in self.query):
                    return self.preprocessor(" ".join(self.query))[0].words

            raise ValueError(f"Invalid query type: {type(self.query)}")

        query = _parse()
        return [x.term for x in query]

    def ppformat(self):
        return " ".join(self.parsed_query)


class PhraseQuery(Query):
    def __init__(self, query, preprocessor=DEFAULT_PREPROCESSOR):
        super().__init__(query, preprocessor)

        self.parsed_query, self.distances = self.parse()

    def parse(self):
        boundary_query = self.query
        processed_query = self.preprocessor(boundary_query.strip('"'))[0].words

        if len(processed_query) == 0:
            return [], []

        distances = [0]
        for i in range(1, len(processed_query)):
            distances.append(
                processed_query[i].start_position
                - processed_query[i - 1].start_position
            )

        result_query = [word.term for word in processed_query]
        return result_query, distances[1:]

    def ppformat(self, level=0):
        return " " * level + '"' + " ".join(self.parsed_query) + '"' + "\n"


class ProximityQuery(Query):
    def __init__(self, query, preprocessor=DEFAULT_PREPROCESSOR):
        super().__init__(query, preprocessor)

        self.regex = re.compile(r"#(\d+)\((\w+),\s*(\w+)\)")
        self.parsed_query, self.distances = self.parse()

    def parse(self):
        match = self.regex.match(self.query)
        if not match:
            raise ValueError(f"Invalid proximity query: {self.query}")

        distance = int(match.group(1))
        terms = [
            self.preprocessor(match.group(2)),
            self.preprocessor(match.group(3)),
        ]
        terms = [
            term[0].words[0].term if len(term) > 0 and len(term[0].words) > 0 else ""
            for term in terms
        ]

        return terms, [distance]

    def ppformat(self, level=0):
        return (
            " " * level
            + f"#{self.distances[0]}({self.parsed_query[0]}, {self.parsed_query[1]})"
            + "\n"
        )


@dataclass
class AND:
    left: Query
    right: Query

    def ppformat(self, level=0):
        out = self.left.ppformat(level + 1)
        out += " " * level + "AND" + "\n"
        out += self.right.ppformat(level + 1)

        return out

    def pprint(self):
        print(self.ppformat())


@dataclass
class OR:
    left: Query
    right: Query

    def ppformat(self, level=0):
        out = self.left.ppformat(level + 1)
        out += " " * level + "OR" + "\n"
        out += self.right.ppformat(level + 1)

        return out

    def pprint(self):
        print(self.ppformat())


@dataclass
class NOT:
    left: Query
    right: Query | None = None

    def ppformat(self, level=0):
        out = self.left.ppformat(level + 1)
        out += " " * level + "NOT" + "\n"
        if self.right:
            out += self.right.ppformat(level + 1)

        return out


class BooleanQuery(Query):
    def __init__(self, query, preprocessor=Preprocessor()):
        super().__init__(query, preprocessor)

        self.precedence = {"NOT": (3, NOT), "AND": (2, AND), "OR": (1, OR)}
        self.operators, self.operands = [], []

    def __iter__(self):
        if not self.operands:
            self.parse()

        query = self.operands[0] if self.operands else None
        if query:
            yield from self.recurse(query)

    def recurse(self, query: Query):
        if isinstance(query, (AND, OR)):
            yield from self.recurse(query.left)
            if query.right:
                yield from self.recurse(query.right)
        elif isinstance(query, NOT):
            yield from self.recurse(query.left)
        elif isinstance(query, (TermQuery, PhraseQuery, ProximityQuery)):
            yield query
        else:
            raise ValueError(f"Invalid query type: {type(query)}")

    def parse(self):
        self.operands, self.operators = [], []

        tokens = re.findall(r'"[^"]+"|#\d+\(\w+,\s*\w+\)|\(|\)|\S+', self.query)
        open_bracket = "("  # )

        def evaluate():
            if not self.operators:
                return

            op = self.operators.pop()
            if op == open_bracket:
                return

            op_class = self.precedence.get(op, (None, None))[1]
            if not op_class:
                return

            right = self.operands.pop()
            if op_class == NOT:
                self.operands.append(op_class(right))
            else:
                left = (
                    self.operands.pop()
                    if self.operands
                    else TermQuery("", self.preprocessor)
                )
                self.operands.append(op_class(left, right))

        for token in tokens:
            if token in self.precedence:
                while (
                    self.operators
                    and self.operators[-1] in self.precedence
                    and self.precedence[self.operators[-1]][0]
                    >= self.precedence[token][0]
                ):
                    evaluate()
                self.operators.append(token)
            elif open_bracket == token:
                self.operators.append(token)
            elif token == ")":
                while self.operators and open_bracket != self.operators[-1]:
                    evaluate()
                if self.operators and open_bracket == self.operators[-1]:
                    self.operators.pop()
            elif token.startswith("#"):
                self.operands.append(
                    ProximityQuery(token, preprocessor=self.preprocessor)
                )
            elif token.startswith('"') and token.endswith('"'):
                self.operands.append(PhraseQuery(token, preprocessor=self.preprocessor))
            else:
                self.operands.append(TermQuery(token, preprocessor=self.preprocessor))

        while self.operators:
            evaluate()

        return self.operands

    def __str__(self):
        return self.__repr__()

    def _ppformat(self, level=0):
        if self.operands:
            out = self.operands[0].ppformat()
        else:
            out = self.__str__()

        return out


if __name__ == "__main__":
    query = BooleanQuery("NOT #1(dog, cat) AND cat OR dog")
    query.parse()

    _query = query.operands.pop()
    _query.pprint()
