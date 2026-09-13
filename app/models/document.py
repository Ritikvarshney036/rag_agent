from dataclasses import dataclass

@dataclass
class Page:
    page_number: int
    text: str
    char_count: int

@dataclass
class Document:
    name: str
    pages: list[Page]

