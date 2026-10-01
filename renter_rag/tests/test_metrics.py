import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics import auroc, credited_gains, facts_found, hit_at_k, mrr, ndcg_at_k, recall_at_k  # noqa: E402

GOLD = {"RTA-s44": 2, "CAV-RENTINC-05": 1}


def test_each_gold_unit_is_credited_once():
    ranked = [["RTA-s44"], ["RTA-s44"], ["CAV-RENTINC-05"]]
    assert credited_gains(ranked, GOLD) == [2, 0, 1]


def test_fixed_chunk_covering_two_units_gets_best_grade():
    assert credited_gains([["RTA-s43", "RTA-s44"]], GOLD) == [2]


def test_recall_and_hit():
    ranked = [["RTA-s1"], ["CAV-RENTINC-05"], ["RTA-s44"]]
    assert recall_at_k(ranked, GOLD, 2) == 0.5
    assert recall_at_k(ranked, GOLD, 3) == 1.0
    assert hit_at_k(ranked, GOLD, 2) == 0.0      # only the grade-1 unit so far
    assert hit_at_k(ranked, GOLD, 3) == 1.0


def test_mrr_uses_first_relevant_rank():
    assert mrr([["x"], ["y"], ["RTA-s44"]], GOLD) == 1 / 3
    assert mrr([["x"]], GOLD) == 0.0


def test_ndcg_perfect_and_duplicates():
    assert ndcg_at_k([["RTA-s44"], ["CAV-RENTINC-05"]], GOLD, 5) == 1.0
    # a duplicate of the same unit must not make the ranking look better than ideal
    dup = ndcg_at_k([["RTA-s44"], ["RTA-s44"], ["CAV-RENTINC-05"]], GOLD, 5)
    assert dup < 1.0
    expected = (3 / math.log2(2) + 1 / math.log2(4)) / (3 / math.log2(2) + 1 / math.log2(3))
    assert math.isclose(dup, expected)


def test_facts_found_regex():
    assert facts_found("You need at least 90 days notice.", ["90 days", "notice"]) == 1.0
    assert facts_found("60 days", ["90 days"]) == 0.0


def test_auroc():
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.5], [0.5]) == 0.5
