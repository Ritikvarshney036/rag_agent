import pymupdf # PyMuPDF
from pathlib import Path
from app.models.document import Document, Page

class PDFParser:
    """
    Responsible for reading a PDF and extracting text page by page.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    def extract_text(self):
        """
        Extract text from every page.

        Returns:
            List[dict]
        """

        document = fitz.open(self.pdf_path)

        pages = []

        for page_number in range(len(document)):
            page = document.load_page(page_number)

            text = page.get_text("text")

            pages.append(
                {
                    "page": page_number + 1,
                    "text": text
                }
            )

        document.close()

        return pages

    def get_total_pages(self):

        document = fitz.open(self.pdf_path)

        total_pages = len(document)

        document.close()

        return total_pages


    def parse(self):

        doc = pymupdf.open(self.pdf_path)

        pages = []

        for index in range(len(doc)):

            page = doc.load_page(index)

            text = page.get_text().strip()

            if not text:
                continue

            pages.append(
                Page(
                    page_number=index + 1,
                    text=text,
                    char_count=len(text)
                )
            )

        doc.close()

        return Document(
            name=self.pdf_path.name,
            pages=pages
        )
            