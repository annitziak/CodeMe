from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    results: list[dict[str, Any]]
    time_taken: float
    total_results: int
    query: str

    def __str__(self):
        return f"Search Result(\n\tresults={self.results[:5]}\n\ttime_taken={self.time_taken}\n\ttotal_results={self.total_results}\n\tquery={self.query}\n)"
