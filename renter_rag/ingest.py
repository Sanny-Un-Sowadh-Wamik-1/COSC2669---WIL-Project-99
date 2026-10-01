"""Build the knowledge base: legislation + CAV guidance -> chunks, in two chunking strategies.

Units vs chunks
---------------
A *unit* is the thing we cite and the thing the test set's gold answers point at:
    RTA-s44        Residential Tenancies Act 1997 s 44
    RTR-r29        Residential Tenancies Regulations 2021 reg 29
    RTR-sch4-c3    ... Schedule 4 clause 3
    CAV-RENTINC-05 the 5th heading on the CAV "Rent increases" page
A *chunk* is what the retriever indexes. Each chunk remembers which unit(s) it came from, so the
two chunking strategies can be scored against the same gold units:

  section : one chunk per legal section / CAV heading. Sections longer than SECTION_MAX_WORDS are
            split at subsection boundaries, and each piece keeps the section heading.
  fixed   : a sliding window of FIXED_WORDS words with FIXED_OVERLAP overlap over each document,
            ignoring structure. Its units are every section the window touches.

Run:  python ingest.py        -> cache/chunks_section.json, cache/chunks_fixed.json
"""

import json
import re

import docx
import yaml

from config import CACHE, FIXED_OVERLAP, FIXED_WORDS, ROOT, SECTION_MAX_WORDS, SOURCES

# Parts of the legislation that are not useful to a renter (and would add noise to retrieval).
SKIP_PARTS = re.compile(r"transitional|validation|forms|savings", re.IGNORECASE)
SKIP_STYLES = ("Side Note", "Stars", "toc", "Amend.", "endnote", "Notes Body")


def _clean(text):
    return " ".join(text.replace("\t", " ").split())


def parse_legislation(doc_meta):
    """Read an authorised .docx and return its sections, in order:
    [{unit, number, title, part, paragraphs: [(style, text)]}]"""
    paragraphs = docx.Document(ROOT / doc_meta["local"]).paragraphs
    sections, current = [], None
    part = division = ""
    schedule = None
    skipping = False
    started = False
    for p in paragraphs:
        style, raw = p.style.name, p.text
        text = _clean(raw)
        if not text or style.startswith(SKIP_STYLES):
            continue
        if style == "Heading - ENDNOTES" or text == "Endnotes" and style.startswith("Heading"):
            break
        if style == "Heading - PART":
            started = True
            part, division = text, ""
            skipping = bool(SKIP_PARTS.search(text))
            m = re.match(r"Schedule (\w+)", text)
            schedule = m.group(1) if m else None
            current = None
            continue
        started = started or style == "Draft Heading 1"   # regulations have no Part headings
        if not started or skipping:
            continue
        if style == "Heading - DIVISION":
            division = text
            continue
        if style == "Draft Heading 1":
            fields = [f.strip() for f in raw.split("\t") if f.strip()]
            if len(fields) < 2:
                continue
            number, title = fields[0], " ".join(fields[1:])
            if schedule:
                unit = f"{doc_meta['id']}-sch{schedule}-c{number}"
                label = f"sch {schedule} cl {number}"
            else:
                prefix = "s" if doc_meta["unit"] == "s" else "r"
                unit = f"{doc_meta['id']}-{prefix}{number}"
                label = f"{doc_meta['unit']} {number}"
            current = {"unit": unit, "label": label, "title": title,
                       "part": " · ".join(x for x in (part, division) if x), "paragraphs": []}
            sections.append(current)
            continue
        if current is not None:
            current["paragraphs"].append((style, text))
    # repealed sections have no body ("* * *") – drop them
    return [s for s in sections if sum(len(t.split()) for _, t in s["paragraphs"]) > 3]


def parse_cav_page(path):
    _, header, body = path.read_text(encoding="utf-8").split("---", 2)
    meta = yaml.safe_load(header)
    sections = []
    for number, block in enumerate([b for b in body.split("\n## ") if b.strip().lstrip("#").strip()], 1):
        heading, _, text = block.strip().lstrip("#").strip().partition("\n")
        sections.append({"unit": f"{meta['doc_id']}-{number:02d}", "label": heading.strip(),
                         "title": heading.strip(), "part": meta["title"],
                         "paragraphs": [("Normal", _clean(line)) for line in text.splitlines() if line.strip()]})
    return meta, sections


def load_documents():
    """All documents as dicts: {id, title, url, date, kind, sections}."""
    sources = yaml.safe_load((ROOT / "sources.yaml").read_text())
    docs = []
    for meta in sources["legislation"]:
        docs.append({"id": meta["id"], "title": meta["title"], "url": meta["page"],
                     "date": f"version {meta['version'].split()[0]}", "kind": "law",
                     "sections": parse_legislation(meta)})
    for path in sorted((SOURCES / "cav").glob("*.md")):
        meta, sections = parse_cav_page(path)
        docs.append({"id": meta["doc_id"], "title": f"CAV: {meta['title']}", "url": meta["url"],
                     "date": f"updated {meta['last_updated']}", "kind": "guide", "sections": sections})
    return docs


def _chunk(doc, section, text, units, n):
    if doc["kind"] == "law":
        citation = f"{doc['title']} {section['label']}"
        heading = f"{section['label']} {section['title']}"
    else:
        citation = f"{doc['title']}: {section['title']}"
        heading = section["title"]
    return {"id": f"{section['unit']}#{n}", "units": units, "doc": doc["id"], "kind": doc["kind"],
            "citation": citation, "heading": heading, "part": section["part"],
            "url": doc["url"], "date": doc["date"], "text": text}


def section_chunks(doc):
    """One chunk per section; long sections split at subsection ('Draft Heading 2') boundaries,
    or at any paragraph if a single subsection is still too long."""
    chunks = []
    for section in doc["sections"]:
        pieces, current, words = [], [], 0
        for style, text in section["paragraphs"]:
            n = len(text.split())
            boundary = style == "Draft Heading 2" or words + n > SECTION_MAX_WORDS * 1.3
            if current and words + n > SECTION_MAX_WORDS and boundary:
                pieces.append(current)
                current, words = [], 0
            current.append(text)
            words += n
        if current:
            pieces.append(current)
        for i, piece in enumerate(pieces, 1):
            chunks.append(_chunk(doc, section, " ".join(piece), [section["unit"]], i))
    return chunks


def fixed_chunks(doc):
    """Sliding window over the whole document; structure is ignored except for remembering
    which section every word came from (so the chunk can still be cited)."""
    words, owner = [], []
    for k, section in enumerate(doc["sections"]):
        for _, text in section["paragraphs"]:
            for w in text.split():
                words.append(w)
                owner.append(k)
    chunks, step = [], FIXED_WORDS - FIXED_OVERLAP
    for n, start in enumerate(range(0, max(len(words) - FIXED_OVERLAP, 1), step), 1):
        span = range(start, min(start + FIXED_WORDS, len(words)))
        sections = list(dict.fromkeys(owner[i] for i in span))
        first = doc["sections"][sections[0]]
        chunk = _chunk(doc, first, " ".join(words[i] for i in span),
                       [doc["sections"][k]["unit"] for k in sections], n)
        chunk["id"] = f"{doc['id']}-w{n:04d}"
        chunks.append(chunk)
    return chunks


def build():
    CACHE.mkdir(exist_ok=True)
    docs = load_documents()
    units = {}
    for doc in docs:
        for s in doc["sections"]:
            units[s["unit"]] = {"citation": _chunk(doc, s, "", [], 0)["citation"], "title": s["title"],
                                "url": doc["url"], "text": " ".join(t for _, t in s["paragraphs"])}
    (CACHE / "units.json").write_text(json.dumps(units, ensure_ascii=False))
    for name, fn in (("section", section_chunks), ("fixed", fixed_chunks)):
        chunks = [c for doc in docs for c in fn(doc)]
        (CACHE / f"chunks_{name}.json").write_text(json.dumps(chunks, ensure_ascii=False))
        print(f"{name:8s}: {len(chunks):5d} chunks")
    for doc in docs[:3]:
        print(f"{doc['title']}: {len(doc['sections'])} sections")
    print(f"CAV guidance: {sum(len(d['sections']) for d in docs[3:])} sections from {len(docs) - 3} pages")
    return docs


if __name__ == "__main__":
    build()
