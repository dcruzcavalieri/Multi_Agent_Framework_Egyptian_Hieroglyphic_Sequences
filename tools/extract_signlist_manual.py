"""Extract the sign list (Anexo I) of Gracia Zamacona, *Manual de egipcio medio* (2nd ed., 2017) by OCR.

The PDF uses fonts without a usable text layer, so pages are rendered as images and read with Tesseract
(Spanish). Table rows are found from the ink of the code column (one code per row), the footnote rule
marks the end of the table, each code cell is re-read with a character whitelist, and the other cells
are rebuilt from the page OCR with words grouped into lines and read left to right.

    python tools/extract_signlist_manual.py <manual.pdf> <out_dir> [--first 108 --last 189]

Writes <out_dir>/signlist_manual_raw.csv with one row per sign:
code, code_ocr, code_check, image (description), ideogram, phonogram, determinative, alternations,
confusions, page, book_page.
OCR of transliterations (ideogram and phonogram columns) is unreliable; the description and
determinative columns are plain Spanish and read well. The book is (c) Archaeopress and C. Gracia
Zamacona: the output is for internal research use and must not be published without permission.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

COLUMNS = ["image", "ideogram", "phonogram", "determinative", "alternations", "confusions"]
SECTIONS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V", "W", "X", "Y", "Z", "Aa"]
HEADER_WORDS = {"Imagen": "image", "Ideograma": "ideogram", "Fono-": "phonogram", "Determinativo": "determinative",
                "Alternan-": "alternations", "Alter-": "alternations", "Confu-": "confusions",
                "Confusio-": "confusions"}
# column left edges at 300 dpi (A5 page, 1750 px wide); refined per page from the header words
DEFAULT_X = {"sign": 270, "image": 400, "ideogram": 620, "phonogram": 850, "determinative": 980,
             "alternations": 1230, "confusions": 1430}
CODE_X0 = 120
WHITELIST = "ABCDEFGHIKLMNOPQRSTUVWXYZabc0123456789"
DIGIT_FOR = {"l": "1", "I": "1", "|": "1", "O": "0", "o": "0", "D": "0", "S": "5", "B": "8", "Z": "2", "g": "9"}


def _read_tsv(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
    out = []
    for r in rows:
        t = (r.get("text") or "").strip()
        if t and r["level"] == "5":
            out.append({"x": int(r["left"]), "y": int(r["top"]), "w": int(r["width"]), "h": int(r["height"]),
                        "t": t})
    return out


def _runs(mask: np.ndarray, gap: int) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    out, start, prev = [], idx[0], idx[0]
    for v in idx[1:]:
        if v - prev > gap:
            out.append((start, prev))
            start = v
        prev = v
    out.append((start, prev))
    return out


def footnote_rule(ink: np.ndarray) -> Optional[int]:
    """y of the short horizontal footnote rule at the left margin, if any."""
    h, w = ink.shape
    for y in range(h // 3, h - 50):
        row = ink[y, CODE_X0:CODE_X0 + 600]
        runs = _runs(row, 1)
        if runs and any(200 <= b - a <= 520 for a, b in runs) and ink[y, CODE_X0 + 650:].sum() < 5 \
                and ink[max(0, y - 6):y - 1, CODE_X0:CODE_X0 + 600].sum() < 10:
            return y
    return None


def _ocr(img: Image.Image, psm: str) -> str:
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.png")
        img.save(p)
        res = subprocess.run(["tesseract", p, "stdout", "--psm", psm, "-c", f"tessedit_char_whitelist={WHITELIST}"],
                             capture_output=True, text=True)
    return res.stdout.strip().replace(" ", "")


def read_code(hi: Image.Image, img: Image.Image, box: Tuple[int, int, int, int]) -> List[str]:
    """Three readings of a code cell (a fourth comes from the full-page OCR): 2x-resolution render (plain and binarised) and the base render
    upscaled. The thin strokes of the book's font break up at 300 dpi, hence the 600 dpi render."""
    out = []
    big = hi.crop(tuple(2 * v for v in box))
    for c in (big, big.point(lambda v: 255 if v > 150 else 0),
              img.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.LANCZOS)):
        pad = Image.new("L", (c.width + 80, c.height + 80), 255)
        pad.paste(c, (40, 40))
        out.append(_ocr(pad, "7"))
    return out


PREFIX_CONFUSIONS = {"I": "1lIT|", "O": "0ODQ", "S": "S5839", "G": "G6", "B": "B8", "Z": "Z2L", "T": "T7"}
NUM = re.compile(r"^([0-9lIOoDSBZgTGQ]{1,3})([a-dA-D]?)$")


def code_candidates(raw: str, section: str) -> List[Tuple[int, str]]:
    """(number, suffix) readings of a raw OCR string as a code of ``section``."""
    s = raw.strip()
    rests = []
    if section == "Aa":
        if s[:2] in ("Aa", "AA", "Ag", "Aq", "Aaa"[:2]):
            rests.append(s[2:])
        elif s[:1] == "A":
            rests.append(s[1:].lstrip("a"))
    else:
        if s[:1] == section or s[:1] in PREFIX_CONFUSIONS.get(section, ""):
            rests.append(s[1:])
        else:                        # prefix misread as another character, or lost
            rests += [s[1:], s]
    out = []
    for rest in rests:
        for k in range(1, min(3, len(rest)) + 1):     # every split into number + optional variant letter
            m = NUM.match(rest[:k])
            suf = rest[k:]
            if not m or m.group(2) or not (suf == "" or re.fullmatch(r"[a-dA-D]", suf)):
                continue
            num = "".join(DIGIT_FOR.get(ch, ch) for ch in m.group(1))
            if num.isdigit() and 0 < int(num) < 200:
                out.append((int(num), suf.upper()))
    return out


def cell_text(words: List[dict]) -> str:
    """Words grouped into lines (by vertical position), each line read left to right, hyphenation joined."""
    lines: List[List[dict]] = []
    for w in sorted(words, key=lambda w: w["y"] + w["h"] / 2):
        cy = w["y"] + w["h"] / 2
        for ln in lines:
            ly = sum(v["y"] + v["h"] / 2 for v in ln) / len(ln)
            if abs(cy - ly) < 14:
                ln.append(w)
                break
        else:
            lines.append([w])
    text = " ".join(" ".join(v["t"] for v in sorted(ln, key=lambda v: v["x"])) for ln in lines)
    return re.sub(r"(\w)-\s+(?=[a-záéíóúñ])", r"\1", text).strip()


def clean_cell(col: str, text: str) -> str:
    """Dashes of empty cells are often read as ']', ':', 'Ñ' ...; single letters are valid only as
    phonetic values."""
    if not text:
        return ""
    if col in ("ideogram", "phonogram"):
        return text if re.search(r"[A-Za-zÀ-ÿ0-9]", text) else "-"
    return text if re.search(r"[A-Za-zÀ-ÿ0-9]{2}", text) else "-"


HEADING = re.compile(r"^(Aa|[A-Z])\.$")


def parse_page(png: str, tsv: str, page: int, section: str, hi_png: str, cache: Optional[dict] = None) -> Tuple[List[dict], Dict[str, List[dict]], str]:
    """Rows of one page, words continuing the previous page's last row, and the section at the page end."""
    img = Image.open(png).convert("L")
    hi = Image.open(hi_png).convert("L")
    ink = np.asarray(img) < 128
    words = _read_tsv(tsv)
    y_end = footnote_rule(ink) or (img.height - 230)
    words = [w for w in words if w["y"] < y_end]
    header_ys = sorted({w["y"] for w in words if w["t"] == "Imagen"})
    # a section heading ("C. Mujeres") is a line starting at the left margin shortly above a header row;
    # its letter is read when OCR got it, otherwise it is the next section of the list
    headings = []
    for hy in header_ys:
        left = sorted((w for w in words if w["x"] < 200 and hy - 260 <= w["y"] <= hy - 60), key=lambda w: w["y"])
        if left:
            m = HEADING.match(left[0]["t"])
            cur = headings[-1][1] if headings else section
            succ = SECTIONS[min(SECTIONS.index(cur) + 1, len(SECTIONS) - 1)]
            headings.append((left[0]["y"], m.group(1) if m and m.group(1) in SECTIONS else succ))
    rows, carry = [], {c: [] for c in COLUMNS}
    for h, hy in enumerate(header_ys):
        xs = dict(DEFAULT_X)
        n_x = 160
        for w in words:
            if abs(w["y"] - hy) < 70:
                key = HEADER_WORDS.get(w["t"])
                if key:   # cell text starts up to ~45 px left of a centred header word
                    xs[key] = w["x"] - (50 if key == "image" else 25)
                if w["t"] in ("Signo", "Sig-"):
                    xs["sign"] = w["x"] - 10
                if w["x"] < 250 and abs(w["y"] - hy) < 15 and w["t"].startswith("N"):
                    n_x = w["x"]
        # the image column's text is left-aligned, often well left of its centred header word
        lefts = sorted(w["x"] for w in words if w["y"] > hy + 85 and xs["sign"] + 60 <= w["x"] <= xs["image"] + 70)
        for x0 in lefts:
            if sum(1 for x in lefts if abs(x - x0) <= 5) >= 3:
                xs["image"] = min(xs["image"], x0 - 8)
                break
        for y, sec in headings:
            if y < hy:
                section = sec
        seg_lo = hy + 85
        nexts = [y for y in header_ys if y > hy] + [y for y, _ in headings if y > hy]
        seg_hi = min(nexts) - 10 if nexts else y_end
        bounds = sorted(((k, v) for k, v in xs.items() if k != "sign"), key=lambda kv: kv[1])

        def col(x):
            name = None
            for n, x0 in bounds:
                if x >= x0:
                    name = n
            return name

        # codes are left-aligned at or a little left of the (italic) "Nº" of the header
        strip = ink[seg_lo:seg_hi, CODE_X0:n_x + 20].sum(axis=1) > 0
        blobs = [(int(a) + seg_lo, int(b) + seg_lo) for a, b in _runs(strip, 8) if 15 <= b - a <= 60]
        first = blobs[0][0] - 12 if blobs else seg_hi
        if h == 0:   # text above the first code continues the previous page's last row
            for w in words:
                if seg_lo - 20 <= w["y"] < first and col(w["x"]) in carry:
                    carry[col(w["x"])].append(w)
        for k, (a, b) in enumerate(blobs):
            key = f"{page}:{a}"
            if cache is not None and key in cache:
                reads = cache[key][:3]
            else:
                reads = read_code(hi, img, (CODE_X0, a - 10, min(n_x + 90, xs["sign"] - 4), b + 8))
                if cache is not None:
                    cache[key] = reads
            page_read = "".join(w["t"] for w in sorted(words, key=lambda w: w["x"])
                                if w["x"] < xs["sign"] - 4 and a - 8 <= w["y"] + w["h"] / 2 <= b + 8)
            reads.append("".join(ch for ch in page_read if ch in WHITELIST))   # the full-page OCR
            y_lo = a - 10
            y_hi = (blobs[k + 1][0] - 10) if k + 1 < len(blobs) else seg_hi
            cells = {c: [] for c in COLUMNS}
            for w in words:   # by the word's vertical centre
                if y_lo <= w["y"] + w["h"] / 2 < y_hi and col(w["x"]) in cells:
                    cells[col(w["x"])].append(w)
            rows.append({"reads": reads, "section": section, "page": page, "y": a, "cells": cells})
    for y, sec in headings:     # heading after the last header (section starting on the next page)
        if not header_ys or y > header_ys[-1]:
            section = sec
    return rows, carry, section


def _trans(a: Tuple[int, str], b: Tuple[int, str]) -> float:
    (n1, s1), (n2, s2) = a, b
    if n2 == n1 and s2 > s1:
        return 0.0                          # variant listed after its base sign
    if n2 == n1 + 1:
        return 0.0 if not s2 else -1.0
    if n1 + 1 < n2 <= n1 + 8:
        return -1.5 * (n2 - n1 - 1)         # the manual skips a few codes
    return -50.0


def resolve_codes(rows: List[dict]) -> None:
    """Choose each row's code by dynamic programming over the rows of a section: OCR votes (three
    readings per cell) against the expected order of the list (n, n+1, variants nA after n)."""
    i = 0
    while i < len(rows):
        sec = rows[i]["section"]
        j = i
        while j < len(rows) and rows[j]["section"] == sec:
            j += 1
        block = rows[i:j]
        votes = []
        for r in block:
            v: Dict[Tuple[int, str], int] = {}
            for raw in r["reads"]:
                for c in set(code_candidates(raw, sec)):
                    v[c] = v.get(c, 0) + 1
            votes.append(v)
        cands = []
        for k in range(len(block)):
            c = set(votes[k])
            if k > 0:
                c |= {(n + 1, "") for n, _ in votes[k - 1]}
            if k + 1 < len(block):
                c |= {(n - 1, "") for n, _ in votes[k + 1] if n > 1}
            c.add((k + 1, ""))               # position in the section as a last resort
            cands.append(sorted(c))
        emit = lambda k, c: 2.0 * votes[k][c] if c in votes[k] else -3.0  # noqa: E731
        best = [{c: (emit(0, c) + _trans((0, ""), c), None) for c in cands[0]}]
        for k in range(1, len(block)):
            layer = {}
            for c in cands[k]:
                prev, sc = max(((p, v[0] + _trans(p, c)) for p, v in best[-1].items()), key=lambda t: t[1])
                layer[c] = (sc + emit(k, c), prev)
            best.append(layer)
        c = max(best[-1], key=lambda x: best[-1][x][0])
        path = [c]
        for k in range(len(block) - 1, 0, -1):
            c = best[k][c][1]
            path.append(c)
        path.reverse()
        prev = (0, "")
        for k, (r, c) in enumerate(zip(block, path)):
            flag = []
            if c not in votes[k]:
                flag.append("inferred")
            if c[0] > prev[0] + 1:
                flag.append(f"gap after {sec}{prev[0]}{prev[1]}")
            r["code"], r["code_check"] = f"{sec}{c[0]}{c[1]}", " ".join(flag)
            r["code_ocr"] = " / ".join(r["reads"])
            prev = c
        i = j
    seen: Dict[str, int] = {}
    for r in rows:
        seen[r["code"]] = seen.get(r["code"], 0) + 1
    for r in rows:
        if seen[r["code"]] > 1:
            r["code_check"] = (r["code_check"] + " duplicate").strip()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pdf")
    ap.add_argument("out_dir")
    ap.add_argument("--first", type=int, default=108)
    ap.add_argument("--last", type=int, default=189)
    ap.add_argument("--book-offset", type=int, default=14, help="PDF page minus printed page number")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)
    pages_dir = os.path.join(args.out_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    hi_dir = os.path.join(args.out_dir, "pages_hi")
    os.makedirs(hi_dir, exist_ok=True)
    if not glob.glob(os.path.join(pages_dir, "*.tsv")):
        subprocess.run(["pdftoppm", "-f", str(args.first), "-l", str(args.last), "-r", str(args.dpi), "-gray",
                        "-png", args.pdf, os.path.join(pages_dir, "p")], check=True)
        for png in sorted(glob.glob(os.path.join(pages_dir, "p-*.png"))):
            subprocess.run(["tesseract", png, png[:-4], "-l", "spa", "--psm", "6", "tsv"],
                           check=True, capture_output=True)
    if not glob.glob(os.path.join(hi_dir, "*.png")):
        subprocess.run(["pdftoppm", "-f", str(args.first), "-l", str(args.last), "-r", str(2 * args.dpi), "-gray",
                        "-png", args.pdf, os.path.join(hi_dir, "p")], check=True)
    cache_path = os.path.join(args.out_dir, "code_reads.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            cache = json.load(fh)
    rows, section = [], "A"
    for tsv in sorted(glob.glob(os.path.join(pages_dir, "p-*.tsv"))):
        page = int(re.search(r"p-(\d+)", tsv).group(1))
        got, carry, section = parse_page(tsv[:-4] + ".png", tsv, page, section,
                                         os.path.join(hi_dir, os.path.basename(tsv)[:-4] + ".png"), cache)
        if rows:
            for c, ws in carry.items():
                rows[-1]["cells"][c] += [dict(w, y=w["y"] + 10 ** 5) for w in ws]
        rows += got
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(cache, fh)
    resolve_codes(rows)
    for r in rows:
        r.update({c: clean_cell(c, cell_text(v)) for c, v in r["cells"].items()})
        r["book_page"] = str(r["page"] - args.book_offset)
    out = os.path.join(args.out_dir, "signlist_manual_raw.csv")
    fields = ["code", "code_check", "code_ocr", "image", "ideogram", "phonogram", "determinative",
              "alternations", "confusions", "page", "book_page", "y"]
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    bad = sum(1 for r in rows if r["code_check"])
    print(f"{len(rows)} rows ({bad} with code_check) -> {out}")


if __name__ == "__main__":
    main()
