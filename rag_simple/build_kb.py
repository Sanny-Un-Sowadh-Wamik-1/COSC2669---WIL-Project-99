"""
STEP 1 - Build the knowledge base

We saved 25 Consumer Affairs Victoria (CAV) renting pages as markdown files in
data/pages/. Each file starts with a small header (between the --- lines) that
records where the page came from, then the page text split by "## " headings.

This script cuts every page into smaller "passages" (one per heading) and gives
each passage an ID we can cite, for example:

    CAV-RENTINC-05  ->  the 5th section of the "Rent increases" page

Why passages and not whole pages? A whole page covers many rules. A passage
covers one rule, so the search finds the exact part that answers the question,
and the answer can point to exactly where it came from.

Run:  python build_kb.py      ->  creates data/passages.json
"""

import json
from pathlib import Path

import yaml

PAGES_FOLDER = Path(__file__).parent / "data" / "pages"
OUTPUT_FILE = Path(__file__).parent / "data" / "passages.json"


def read_page(path):
    """Split one markdown file into its header (a dict) and its body (text)."""
    text = path.read_text(encoding="utf-8")
    _, header_text, body = text.split("---", 2)   # the header sits between the first two ---
    header = yaml.safe_load(header_text)
    return header, body


def split_into_passages(header, body):
    """Cut the page body at every '## ' heading. Each section becomes one passage."""
    passages = []
    sections = body.split("\n## ")
    number = 0
    for section in sections:
        section = section.strip().lstrip("#").strip()
        if not section:
            continue
        heading, _, section_text = section.partition("\n")
        number += 1
        passages.append({
            "id": f"{header['doc_id']}-{number:02d}",     # e.g. CAV-RENTINC-05
            "page_title": header["title"],
            "heading": heading.strip(),
            "text": " ".join(section_text.split()),      # tidy up line breaks and spaces
            "url": header["url"],
            "last_updated": str(header["last_updated"]),
        })
    return passages


def main():
    all_passages = []
    for path in sorted(PAGES_FOLDER.glob("*.md")):
        header, body = read_page(path)
        all_passages.extend(split_into_passages(header, body))

    OUTPUT_FILE.write_text(json.dumps(all_passages, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(all_passages)} passages from {len(list(PAGES_FOLDER.glob('*.md')))} pages "
          f"to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
