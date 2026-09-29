import json
import re
from pathlib import Path

from ingestion import extract_pages


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def clean_text(text: str) -> str:
    """Clean extracted PDF text while preserving meaningful content."""

    text = text.replace("\x00", " ")

    # Normalize excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping character-based chunks."""

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def build_chunks(pdf_path: str) -> list[dict]:
    """Extract pages and convert them into searchable chunks."""

    pages = extract_pages(pdf_path)

    chunks = []

    for page in pages:
        page_number = page["page"]
        source = page["source"]

        text = clean_text(page["text"])

        page_chunks = chunk_text(text)

        for chunk_number, chunk in enumerate(page_chunks, start=1):
            chunks.append(
                {
                    "chunk_id": f"page_{page_number}_chunk_{chunk_number}",
                    "page": page_number,
                    "source": source,
                    "text": chunk,
                }
            )

    return chunks


def save_chunks(chunks: list[dict], output_path: str) -> None:
    """Save chunks as JSONL."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    pdf_path = "data/raw/constitution_2024.pdf"
    output_path = "data/processed/chunks.jsonl"

    chunks = build_chunks(pdf_path)

    save_chunks(chunks, output_path)

    print(f"Total chunks created: {len(chunks)}")
    print(f"Saved to: {output_path}")

    if chunks:
        print("\nFirst chunk:")
        print(chunks[0]["text"][:1000])

        print("\nFirst chunk metadata:")
        print(
            {
                "chunk_id": chunks[0]["chunk_id"],
                "page": chunks[0]["page"],
                "source": chunks[0]["source"],
            }
        )