"""Build a sign lexicon for the corpus from the OCR'd sign list of Gracia Zamacona's Manual (2017).

    python tools/build_lexicon_manual.py <signlist_manual_raw.csv> <out_dir>

Writes, for every sign code of the corpus (data/raw/extractrion_for_agents.xlsx):
* <out_dir>/lexicon_manual.csv: the matched entry of the manual (exact code, or the base sign for
  variants written as N9VARA, P8h ...), with "variante del anterior" rows resolved to the sign they vary;
* <out_dir>/revision_lexico_manual.xlsx: a review sheet in Spanish for the expert (one row per corpus
  sign, the OCR fields, and empty columns for corrections).

The manual is (c) Archaeopress and C. Gracia Zamacona. Both outputs are internal research material:
keep them out of the public repository unless the author agrees.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from egydet.gold.build import load_corpus  # noqa: E402
from egydet.gold.io import load_gold  # noqa: E402
from egydet.kb import load_kb  # noqa: E402
from egydet.mdc.tokenizer import tokenize  # noqa: E402

FIELDS = ["image", "ideogram", "phonogram", "determinative", "alternations", "confusions"]
CODE = re.compile(r"^(Aa|[A-Z])(\d+)([A-Z]?)$")
VARIANT = re.compile(r"^(variante( antigua)? del anterior|idem)\b", re.I)


def _empty(v: str) -> bool:
    return v.strip() in ("", "-", "—", "–")


def load_manual(path: str) -> Dict[str, dict]:
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: Dict[str, dict] = {}
    prev: Optional[dict] = None
    for r in rows:
        r = dict(r)
        r["code"] = r["code"].upper() if not r["code"].startswith("Aa") else "Aa" + r["code"][2:].upper()
        r["variant_of"] = ""
        if prev is not None and VARIANT.match(r["image"].strip()):
            r["variant_of"] = prev["variant_of"] or prev["code"]
            r["image"] = f"{r['image'].strip()} ({prev['code']}: {prev['image'].strip()})"
            for f in FIELDS[1:]:
                if _empty(r[f]):
                    r[f] = prev[f]
        out.setdefault(r["code"], r)          # first occurrence wins on duplicated codes
        if not r["variant_of"]:
            prev = r                          # chains of variants point to the last base sign
    return out


def base_code(code: str) -> Optional[str]:
    """N9VARA -> N9, P8h -> P8, O29v -> O29, V31A -> V31."""
    m = re.match(r"^(Aa|[A-Z])(\d+)", code)
    return f"{m.group(1)}{m.group(2)}" if m else None


def match(code: str, manual: Dict[str, dict]):
    up = code if code.startswith("Aa") else code.upper()
    if up in manual:
        return manual[up], "exact"
    if "&" in code:
        return None, "ligature"
    b = base_code(code)
    if b and b != up and b in manual:
        return manual[b], "base sign"
    return None, "not in manual"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("raw_csv")
    ap.add_argument("out_dir")
    args = ap.parse_args(argv)
    manual = load_manual(args.raw_csv)
    corpus = load_corpus(os.path.join(ROOT, "data", "raw", "extractrion_for_agents.xlsx"))
    n_corpus = Counter(t.text for m in corpus["mdc"] for t in tokenize(m))
    gold = load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl"))
    n_gold = Counter(t for r in gold for t in r["tokens"])
    n_det = Counter(r["tokens"][i] for r in gold for i in r["det_idx"])
    kb = load_kb()
    lex = []
    for code, n in sorted(n_corpus.items(), key=lambda kv: (-kv[1], kv[0])):
        if not re.match(r"^(Aa|[A-Z])\d", code):
            continue                          # punctuation, markup and unknown codes
        e, how = match(code, manual)
        row = {"corpus_code": code, "n_corpus": n, "n_gold": n_gold.get(code, 0), "n_gold_det": n_det.get(code, 0),
               "in_kb": code in kb.signs, "match": how, "manual_code": e["code"] if e else "",
               "variant_of": e["variant_of"] if e else "", "book_page": e["book_page"] if e else "",
               "code_check": e["code_check"] if e else ""}
        for f in FIELDS:
            row[f] = (e[f] if e else "").strip()
        lex.append(row)
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "lexicon_manual.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lex[0]))
        w.writeheader()
        w.writerows(lex)
    write_review(lex, os.path.join(args.out_dir, "revision_lexico_manual.xlsx"))
    c = Counter(r["match"] for r in lex)
    print(f"{len(lex)} corpus signs: {dict(c)} -> {out_csv}")


def write_review(lex: List[dict], path: str) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ins = wb.active
    ins.title = "Instrucciones"
    text = [
        "Revisión del léxico de signos extraído del Anexo I del Manual de egipcio medio (2017)",
        "",
        "Cada fila es un signo (código Gardiner) que aparece en el corpus de 533 secuencias. Las columnas "
        "C a K proceden del Anexo I leído por OCR y pueden contener errores, sobre todo en las "
        "transliteraciones.",
        "El objetivo es dar a los agentes una descripción breve de cada signo y de su uso como "
        "determinativo. Lo más importante es la columna 'Determinativo' (campo semántico).",
        "",
        "Por favor, para cada fila:",
        "1. Columna L: indique si la entrada es correcta para el signo ('sí', 'no' o 'corregida').",
        "2. Columnas M y N: si hace falta, escriba la descripción o el uso como determinativo correctos.",
        "3. Columna O: cualquier comentario (p. ej. si el código del corpus corresponde a otro signo).",
        "",
        "Las filas con 'base sign' en la columna 'Correspondencia' son variantes del corpus (p. ej. N9VARA) "
        "a las que se ha asignado la entrada del signo base. Las filas sin correspondencia no figuran en el "
        "Anexo I con ese código.",
        "Las filas están ordenadas por frecuencia en el corpus; las primeras son las más importantes.",
        "Las celdas amarillas (columnas L a O) son las únicas que hay que rellenar.",
        "",
        "Ejemplo de fila rellenada (formato esperado):",
        "  Código D54 · Determinativo (OCR): 'movimiento' · ¿Correcto?: corregida · Descripción corregida: "
        "'Piernas en movimiento' · Determinativo corregido: 'movimiento, desplazamiento' · "
        "Comentario: 'también con verbos de llegar'",
        "",
        "Material de uso interno: el contenido procede del Manual (© Archaeopress y C. Gracia Zamacona).",
    ]
    for i, t in enumerate(text, 1):
        ins.cell(row=i, column=1, value=t).alignment = Alignment(wrap_text=True)
    for row in ins.iter_rows():
        for c in row:
            c.font = Font(name="Arial", size=10)
    ins["A1"].font = Font(name="Arial", bold=True, size=13)
    ins.column_dimensions["A"].width = 110

    ws = wb.create_sheet("Revisión")
    head = ["Código corpus", "Frecuencia corpus", "Código Manual", "Correspondencia", "Página Manual",
            "Imagen (descripción)", "Ideograma", "Fonograma", "Determinativo", "Alternancias", "Confusiones",
            "¿Correcto?", "Descripción corregida", "Determinativo corregido", "Comentario"]
    ws.append(head)
    how_es = {"exact": "exacta", "base sign": "base sign", "not in manual": "sin correspondencia",
              "ligature": "ligadura"}
    for r in lex:
        page = r["book_page"]
        ws.append([r["corpus_code"], r["n_corpus"], r["manual_code"] + (f" (var. de {r['variant_of']})"
                                                                         if r["variant_of"] else ""),
                   how_es[r["match"]], int(page) if page else None, r["image"], r["ideogram"], r["phonogram"],
                   r["determinative"], r["alternations"], r["confusions"], None, None, None, None])
    widths = [13, 10, 16, 16, 9, 34, 26, 12, 34, 16, 16, 12, 30, 30, 30]
    for i, wdt in enumerate(widths, 1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = wdt
    fill = PatternFill("solid", fgColor="DDEBF7")
    inp = PatternFill("solid", fgColor="FFF2CC")
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font = Font(name="Arial", size=10)
    for c in ws[1]:
        c.font = Font(name="Arial", bold=True, size=10)
        c.fill = fill
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for c in row[11:]:
            c.fill = inp
    dv = DataValidation(type="list", formula1='"sí,no,corregida"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"L2:L{ws.max_row}")
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:O{ws.max_row}"
    wb.save(path)


if __name__ == "__main__":
    main()
