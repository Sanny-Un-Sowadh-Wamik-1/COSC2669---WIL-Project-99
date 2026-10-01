"""Retrieval and answer metrics. Pure functions, unit-tested in tests/test_metrics.py.

Retrieval is scored at the level of *units* (sections / CAV headings) so that the two chunking
strategies are comparable: a retrieved chunk earns the grade of the best gold unit it covers, and
each gold unit is credited only once (the first time it appears), so five chunks of the same
section do not count five times.
"""

import math
import re


def credited_gains(ranked_units, gold):
    """ranked_units: list (one entry per retrieved chunk) of lists of unit ids.
    Returns the gain earned at each rank, crediting every gold unit at most once."""
    seen, gains = set(), []
    for units in ranked_units:
        new = [u for u in units if u in gold and u not in seen]
        seen.update(new)
        gains.append(max((gold[u] for u in new), default=0))
    return gains


def recall_at_k(ranked_units, gold, k):
    found = {u for units in ranked_units[:k] for u in units if u in gold}
    return len(found) / len(gold) if gold else 0.0


def hit_at_k(ranked_units, gold, k):
    """1 if any unit that *answers* the question (grade 2) is in the top k."""
    best = max(gold.values())
    return float(any(gold.get(u) == best for units in ranked_units[:k] for u in units))


def mrr(ranked_units, gold):
    for rank, units in enumerate(ranked_units, 1):
        if any(u in gold for u in units):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_units, gold, k):
    gains = credited_gains(ranked_units[:k], gold)
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(gold.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def facts_found(answer, facts):
    """Fraction of the key-fact patterns present in the answer (case-insensitive regex)."""
    if not facts:
        return 0.0
    return sum(bool(re.search(f, answer, re.IGNORECASE)) for f in facts) / len(facts)


def auroc(positive_scores, negative_scores):
    """Probability a random positive scores higher than a random negative (ties count half).
    Used to compare confidence signals: how well does each one separate answerable questions
    from unanswerable ones?"""
    if not positive_scores or not negative_scores:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positive_scores for n in negative_scores)
    return wins / (len(positive_scores) * len(negative_scores))


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None
