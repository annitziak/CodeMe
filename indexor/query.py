import re

from dataclasses import dataclass

from preprocessing.preprocessor import Preprocessor

DEFAULT_PREPROCESSOR = Preprocessor(parser_kwargs={"parser_type": "raw"})


class Query:
    def __init__(self, query, preprocessor=DEFAULT_PREPROCESSOR):
        self.query = query
        self.preprocessor = preprocessor

    def parse(self):
        return self.query

    def pprint(self, level=0):
        print(self.ppformat(level=level))

    def ppformat(self, level=0):
        return " " * level + self.query

    def __str__(self):
        return self.ppformat()


class TermQuery(Query):
    def __init__(self, query, preprocessor=DEFAULT_PREPROCESSOR):
        super().__init__(query, preprocessor)

        self.parsed_query = self.parse()

    def parse(self):
        return self.preprocessor(self.query)


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


@dataclass
class AND:
    left: Query
    right: Query

    def ppformat(self, level=0):
        out = self.left.ppformat(level + 1)
        out += print(" " * level + "AND")
        out += self.right.ppformt(level + 1)

        return out

    def pprint(self):
        print(self.ppformat())


@dataclass
class OR:
    left: Query
    right: Query

    def ppformat(self, level=0):
        out = self.left.ppformat(level + 1)
        out += " " * level + "OR"
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
        out += " " * level + "NOT"
        if self.right:
            out += self.right.ppformat(level + 1)

        return out


class BooleanQuery(Query):
    def __init__(self, query, preprocessor=Preprocessor()):
        super().__init__(query, preprocessor)

        self.precedence = {"NOT": (3, NOT), "AND": (2, AND), "OR": (1, OR)}
        self.operators, self.operands = [], []

    def __iter__(self):
        query = self.operands.pop() if self.operands else self.parse().pop()
        return self.recurse(query)

    def recurse(self, query: Query):
        if isinstance(query, (AND, OR, NOT)):
            if query.right:
                yield list(self.recurse(query.right))
        elif isinstance(query, TermQuery):
            yield query
        elif isinstance(query, PhraseQuery):
            yield query
        elif isinstance(query, ProximityQuery):
            yield query
        else:
            raise ValueError(f"Invalid query type: {type(query)}")

    def parse(self):
        self.operands, self.operators = [], []

        tokens = re.findall(r'"[^"]+"|#\d+\(\w+,\s*\w+\)|\(|\)|\S+', self.query)
        open_bracket = r"("  # )

        def evaluate():
            if self.operators:
                op = self.precedence[self.operators.pop()][1]
                right = self.operands.pop()
                left = self.operands.pop() if self.operands else TermQuery("")
                self.operands.append(op(left, right))

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
                self.operators.pop()
            elif token.startswith("#"):
                self.operands.append(ProximityQuery(token))
            elif token.startswith('"') and token.endswith('"'):
                self.operands.append(PhraseQuery(token))
            else:
                self.operands.append(TermQuery(token))
        while self.operators:
            evaluate()

        return self.operands

    def __str__(self):
        return self.__repr__()

    def ppformat(self):
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
