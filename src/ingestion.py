from pathlib import Path
from pypdf import PdfReader


def extract_pages(pdf_path: str) -> list[dict]:
    """
    Extract text from every page of the Constitution PDF.

    Returns:
        A list of dictionaries containing page number and page text.
    """
    pdf = Path(pdf_path)

    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    reader = PdfReader(str(pdf))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        text = text.strip()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                    "source": pdf.name,
                }
            )

    return pages


if __name__ == "__main__":
    pdf_path = "data/raw/constitution_2024.pdf"

    pages = extract_pages(pdf_path)

    print(f"Total extracted pages: {len(pages)}")

    if pages:
        print("\nFirst page preview:")
        print(pages[0]["text"][:1000])