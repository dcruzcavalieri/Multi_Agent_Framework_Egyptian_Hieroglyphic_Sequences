"""Read the expert-annotated DOCX files and recover determinative *positions*.

Format of the annotation files (one block per sequence):

    Archivo de imagen: ...
    Sequência: E34-N35-**O31**-O4-...          <- determinatives in bold (expert judgement)
    Determinativos: D54, N35, M17               <- system output reviewed by the expert;
                                                   signs judged wrong are coloured red (FF0000)
    Especialista: <translation>

Bold is a run-level property, so every character of the sequence carries its own bold flag and the
position of each bold sign is recovered exactly, even when the same sign occurs several times.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from egydet.mdc.tokenizer import tokenize

SEQ_PREFIX = "Sequência"
PRED_PREFIX = "Determinativos"
TRANSL_PREFIX = "Especialista"
IMAGE_PREFIX = "Archivo de imagen"
RED = "FF0000"


@dataclass
class AnnotatedSequence:
    source_file: str
    paragraph_index: int
    mdc: str                                   # normalised (no whitespace)
    det_idx: List[int]                         # sign-token indices fully in bold
    partial_bold_idx: List[int] = field(default_factory=list)   # tokens only partly bold
    system_pred: List[str] = field(default_factory=list)        # "Determinativos:" line
    system_pred_rejected: List[str] = field(default_factory=list)  # red items on that line
    expert_translation: str = ""
    image_file: str = ""


def _run_color(run) -> Optional[str]:
    color = run.font.color
    if color is None or color.type is None:
        return None
    return str(color.rgb)


def _after_prefix(text: str, prefix: str) -> Optional[int]:
    """Index of the first character after '<prefix>:' or None."""
    stripped = text.lstrip()
    if not stripped.startswith(prefix):
        return None
    colon = text.find(":", text.find(prefix) + len(prefix))
    return None if colon < 0 else colon + 1


def _parse_sequence_paragraph(para):
    """Return (normalised_mdc, bold_mask) for a 'Sequência:' paragraph."""
    chars, bold = [], []
    for run in para.runs:
        for ch in run.text:
            chars.append(ch)
            bold.append(bool(run.bold))
    text = "".join(chars)
    cut = _after_prefix(text, SEQ_PREFIX)
    if cut is None:
        return None, None
    seq_chars, seq_bold = [], []
    for ch, b in zip(text[cut:], bold[cut:]):
        if ch.isspace():
            continue
        seq_chars.append(ch)
        seq_bold.append(b)
    return "".join(seq_chars), seq_bold


def _parse_prediction_paragraph(para):
    """Return (all_predicted_signs, red_signs) from a 'Determinativos:' paragraph."""
    chars, red = [], []
    for run in para.runs:
        is_red = _run_color(run) == RED
        for ch in run.text:
            chars.append(ch)
            red.append(is_red)
    text = "".join(chars)
    cut = _after_prefix(text, PRED_PREFIX)
    if cut is None:
        return [], []
    items, rejected = [], []
    current, current_red = "", False
    for ch, is_red in zip(text[cut:] + ",", red[cut:] + [False]):
        if ch in ",;":
            sign = current.strip()
            if sign:
                items.append(sign)
                if current_red:
                    rejected.append(sign)
            current, current_red = "", False
        else:
            current += ch
            if is_red and not ch.isspace():
                current_red = True
    return items, rejected


def parse_docx(path: str) -> List[AnnotatedSequence]:
    from docx import Document  # python-docx

    doc = Document(path)
    out: List[AnnotatedSequence] = []
    pending_image = ""
    for p_idx, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text.startswith(IMAGE_PREFIX):
            cut = _after_prefix(para.text, IMAGE_PREFIX)
            pending_image = para.text[cut:].strip() if cut else ""
        elif text.startswith(SEQ_PREFIX):
            mdc, mask = _parse_sequence_paragraph(para)
            if not mdc:
                continue
            det_idx, partial = [], []
            for tok in tokenize(mdc):
                flags = mask[tok.start:tok.end]
                if all(flags):
                    det_idx.append(tok.index)
                elif any(flags):
                    partial.append(tok.index)
            out.append(AnnotatedSequence(path, p_idx, mdc, det_idx, partial, image_file=pending_image))
            pending_image = ""
        elif text.startswith(PRED_PREFIX) and out and not out[-1].system_pred:
            out[-1].system_pred, out[-1].system_pred_rejected = _parse_prediction_paragraph(para)
        elif text.startswith(TRANSL_PREFIX) and out and not out[-1].expert_translation:
            cut = _after_prefix(para.text, TRANSL_PREFIX)
            out[-1].expert_translation = para.text[cut:].strip() if cut else ""
    return out
