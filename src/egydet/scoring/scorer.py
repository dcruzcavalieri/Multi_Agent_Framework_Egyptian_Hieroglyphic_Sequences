"""Deterministic scorer and span selection (port of v1.47 ``_score_candidate_p``, ``_score_span_p``,
``_select_dets_parametric`` and ``validate_semantic_context``).

Implementation notes:
* Parameters are passed explicitly (no module-level state). The strict-evidence regime of the
  semantic-context check is read from ``kb.strict_context_signs``.
* Indices are sign indices.
The arithmetic and its order are kept identical so that predictions match v1.47 bit for bit on the
rule-only path (see tests/test_rule_path.py).
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

from egydet.kb import KnowledgeBase, context_tiers
from egydet.rules.baseline import Candidate
from egydet.rules.structure import Structure

BEAM = 6


def semantic_context(sign: str, signs: Tuple[str, ...], kb: KnowledgeBase, semantic_hints: str = "") -> tuple:
    """(has_context, strength) for context-critical signs; (True, 1.0) for all other signs."""
    fld = kb.context_field(sign)
    if not fld:
        return True, 1.0
    field_items = kb.semantic_fields.get(fld, frozenset())
    matches = sum(1 for t in signs if t in field_items)
    if semantic_hints:
        low = str(semantic_hints).lower()
        for kw in field_items:
            if kw.lower() in low:
                matches += 2
                break
    if sign in kb.strict_context_signs:
        strength = min(1.0, matches * 0.25)
        return strength >= 0.6, strength
    strength = min(1.0, matches * 0.3)
    return strength >= 0.5, strength


def score_candidate(c: Candidate, signs: Tuple[str, ...], p: Mapping[str, float], kb: KnowledgeBase) -> float:
    tok = c.sign
    src_n = len(c.sources)
    if tok in kb.tier("phonogram_strict"):
        return 0.0

    if c.conf_sem > 0.0 or c.conf_struct > 0.0:
        w_s = p.get("w_sem_llm", p.get("w_sem", 0.0))
        w_t = p.get("w_struct_llm", p.get("w_struct", 0.0))
        w_r = p.get("w_rule_llm", p.get("w_rule", 1.0))
    else:
        w_s, w_t, w_r = 0.0, 0.0, p["w_rule"]
    score = w_s * c.conf_sem + w_t * c.conf_struct + w_r * c.conf_rule

    if src_n >= 2:
        score += p["bonus_2src"]
    if src_n >= 3:
        score += p["bonus_3src"]

    if tok in kb.tier("phonogram_primary"):
        score *= p["penalty_phonogram_primary_mult"]

    if any(tok in kb.tier(t) for t in context_tiers()):
        has_ctx, ctx_str = semantic_context(tok, signs, kb, c.semantic_hints)
        threshold = p.get(f"thresh_{tok}", p["thresh_default_context"])
        if not has_ctx:
            if score < threshold:
                return 0.0
            score *= p["penalty_no_context_mult"]
        else:
            score *= (1.0 + ctx_str * p["context_strength_boost"])

    if tok in kb.recently_added:
        if tok in kb.recently_critical:
            score *= p["penalty_recently_critical"]
        elif tok in kb.recently_minor:
            score *= p["penalty_recently_minor"]
        else:
            score *= p["penalty_recently_other"]

    if tok in kb.tier("reliable"):
        score += p["bonus_reliable"]
    if tok in kb.tier("plural_functional"):
        if c.is_plural_marker or c.is_near_end:
            score += p["bonus_plural_marker"]
        if c.is_plural_head:
            score += p["bonus_plural_head"]

    if c.is_final:
        score += p["bonus_final"]
    if c.is_group_end and not c.is_final:
        score += p["bonus_group_end"]
    if c.is_global_final:
        score += p["bonus_global_final"]
    if c.is_near_end and not c.is_final:
        score += p["bonus_near_end"]

    if tok in kb.strong_determinatives and tok not in kb.tier("reliable"):
        score += p["bonus_strong_det"]

    if c.is_phonetic_complement_likely:
        score -= p["penalty_phonetic_complement"] if src_n <= 1 else p["penalty_phonetic_complement"] * 0.4

    if tok in kb.tier("phonogram_primary") and not c.is_final:
        score -= p["penalty_phonogram_nonfinal"] if src_n <= 1 else p["penalty_phonogram_nonfinal"] * 0.5

    return max(0.0, min(1.0, score))


def score_span(span: Tuple[int, ...], signs: Tuple[str, ...], occ: Mapping[int, float],
               p: Mapping[str, float], kb: KnowledgeBase) -> float:
    if not span:
        return 0.0
    toks = [signs[i] for i in span]
    s = sum(occ.get(i, 0.0) + (p["bonus_strong_det_span"] if signs[i] in kb.strong_determinatives else 0.0)
            for i in span)
    if any(t in kb.plural_markers for t in toks):
        s += p["bonus_plural_span"]
    if any(t in kb.final_location_signs for t in toks):
        s += p["bonus_location_span"]
    if any(t in kb.location_modifier_signs for t in toks) and any(t in kb.final_location_signs for t in toks):
        s += p["bonus_location_modifier_span"]
    if any(t in kb.contextual_only_signs for t in toks):
        s -= p["penalty_contextual_span"]
    return max(0.0, min(3.0, s))


def select(st: Structure, pool: Dict[int, Candidate], p: Mapping[str, float], kb: KnowledgeBase,
           beam_width: int = BEAM) -> List[int]:
    """Beam search over per-word suffix spans; returns the selected sign indices (in order)."""
    occ = {i: score_candidate(c, st.signs, p, kb) for i, c in pool.items()}
    beam = [(0.0, [])]
    never = kb.never_determinative
    for cands in st.span_candidates:
        new_beam = []
        for score, dets in beam:
            for span in cands:
                sp = span
                if any(st.signs[i] in never for i in sp):
                    sp = ()
                new_beam.append((score + score_span(sp, st.signs, occ, p, kb), dets + list(sp)))
        new_beam.sort(key=lambda x: -x[0])
        beam = new_beam[:beam_width]
    return beam[0][1] if beam else []


def select_ranked(st: Structure, pool: Dict[int, Candidate], p: Mapping[str, float], kb: KnowledgeBase,
                  beam_width: int = 10, k: int = 4) -> List[Tuple[float, List[int]]]:
    """Up to ``k`` distinct determinative sets from the final beam, best first (main + alternatives).

    Same search as ``select`` with a wider beam; spans with zero score are skipped (they would only
    produce copies of a set plus signs without evidence), and sets are deduplicated by their indices.
    """
    occ = {i: score_candidate(c, st.signs, p, kb) for i, c in pool.items()}
    beam = [(0.0, [])]
    never = kb.never_determinative
    for cands in st.span_candidates:
        new_beam = []
        for score, dets in beam:
            for span in cands:
                sp = () if any(st.signs[i] in never for i in span) else span
                gain = score_span(sp, st.signs, occ, p, kb)
                if sp and gain <= 0.0:
                    continue        # a span without evidence only duplicates the empty choice
                new_beam.append((score + gain, dets + list(sp)))
        new_beam.sort(key=lambda x: -x[0])
        beam = new_beam[:beam_width]
    out, seen = [], set()
    for score, dets in beam:
        key = tuple(sorted(dets))
        if key in seen:
            continue
        seen.add(key)
        out.append((score, sorted(dets)))
        if len(out) >= k:
            break
    return out
