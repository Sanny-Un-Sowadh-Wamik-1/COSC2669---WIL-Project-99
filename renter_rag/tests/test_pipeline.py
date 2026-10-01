import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest import fixed_chunks, parse_cav_page, section_chunks  # noqa: E402
from pipeline import cited_numbers, is_other_jurisdiction, says_dont_know  # noqa: E402
from retrieval import is_other_tenure  # noqa: E402

CAV = Path(__file__).parent.parent / "data" / "sources" / "cav" / "rent-increases.md"


def test_other_jurisdictions():
    assert is_other_jurisdiction("What is the maximum bond in NSW?")
    assert is_other_jurisdiction("Is there a cap on rent increases in the ACT?")
    assert is_other_jurisdiction("rules for renting in Queensland")
    assert not is_other_jurisdiction("What does the Act say about rent increases?")
    assert not is_other_jurisdiction("I was late with rent")


def test_dont_know_and_citations():
    assert says_dont_know("I don't know.")
    assert says_dont_know("  I do not know")
    assert not says_dont_know("You get 90 days notice [1].")
    assert cited_numbers("A [2]. B [1][2]. C [9].", k=5) == [2, 1]


def _doc():
    meta, sections = parse_cav_page(CAV)
    return {"id": meta["doc_id"], "title": meta["title"], "url": meta["url"], "date": "x", "kind": "guide",
            "sections": sections}


def test_cav_sections_keep_ids_and_headings():
    doc = _doc()
    chunks = section_chunks(doc)
    assert chunks[0]["units"] == ["CAV-RENTINC-01"]
    assert chunks[0]["heading"] == "Rules for increasing rent"


def test_fixed_chunks_overlap_and_remember_units():
    chunks = fixed_chunks(_doc())
    first, second = chunks[0]["text"].split(), chunks[1]["text"].split()
    assert first[150:200] == second[:50]                      # 50-word overlap
    assert all(c["units"] for c in chunks)


def test_tenure_flag():
    base = {"doc": "RTA", "part": "Part 2—Residential tenancies", "heading": "s 44 Rent increases"}
    assert not is_other_tenure(base)
    assert is_other_tenure({**base, "part": "Part 3—Rooming houses—Residency rights and duties"})
    assert is_other_tenure({**base, "doc": "RHS"})
