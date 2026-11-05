"""Expert evaluation workbook (Spanish) for the alternatives and justifications of a Phase 1 run.

    python -m egydet.legacy_bridge.expert_sheet runs/<collect-run> [--n 30] [--seed 2026]

Writes into the run directory:
* ``evaluacion_expertos.xlsx``   the workbook to send (one copy per evaluator);
* ``evaluacion_clave.csv``       sample key with the stratum of every row (keep it, do not send it).

The sample is stratified by outcome of the published v1.47 output (positional): main set exact; main
wrong but a published alternative better; main wrong and no better alternative; justification citing
sign codes absent from the sequence. Rows are shuffled so that the stratum is not visible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import Dict, List

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from egydet.eval.metrics import score_positions
from egydet.gold.io import load_gold
from egydet.kb import ROOT
from egydet.legacy_bridge.ablation import load_records, parse_answer
from egydet.legacy_bridge.alternatives import SIGN_CODE, _interp_idx
from egydet.mdc import tokenize_with_separators

STRATA = (("A_principal_correcta", 8), ("B_alternativa_mejor", 7), ("C_sin_alternativa_mejor", 9),
          ("D_codigos_ajenos", 6))
FONT = "Arial"
FILL_INPUT = PatternFill("solid", fgColor="FFF2CC")
FILL_HEAD = PatternFill("solid", fgColor="D9E1F2")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def _legacy_label(legacy: List[str], sign_idx: List[int]) -> str:
    """'A1@12, Z1@14' with the legacy token index used in the system texts."""
    pos = [ti for ti, t in enumerate(legacy) if t not in ("-", ":", "*")]
    return ", ".join(f"{legacy[pos[i]]}@{pos[i]}" for i in sorted(sign_idx)) or "(ninguno)"


def _indexed_sequence(legacy: List[str], marked: List[int]) -> str:
    pos = [ti for ti, t in enumerate(legacy) if t not in ("-", ":", "*")]
    m = {pos[i] for i in marked}
    return " ".join((f"[{t}@{ti}]" if ti in m else f"{t}@{ti}") for ti, t in enumerate(legacy) if ti in set(pos))


def _alts_text(sj: dict, legacy: List[str]) -> str:
    out = []
    for k, alt in enumerate((sj.get("alternative_interpretations") or [])[:3], 1):
        idx = _interp_idx(alt, legacy)
        persp = str(alt.get("perspective") or "").strip()
        reason = str(alt.get("reason") or "").strip()
        out.append(f"Alternativa {k} ({persp or 's/p'}): {_legacy_label(legacy, idx)}\n   Condición: {reason or '(sin condición)'}")
    return "\n".join(out) or "(sin alternativas)"


def build_rows(run_dir: str) -> List[dict]:
    gold = {r["id"]: r for r in load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl"))}
    rows = []
    for rec in load_records(run_dir):
        g = gold[rec["id"]]
        legacy = tokenize_with_separators(rec["mdc"])
        sj = rec["state"].get("synthesizer_json") or {}
        mi = sj.get("main_interpretation") or {}
        main = sorted({i for i in rec["final_idx"] if i is not None})
        alts = [_interp_idx(a, legacy) for a in (sj.get("alternative_interpretations") or [])[:3]]
        f_main = score_positions(main, g["det_idx"]).f1
        main_exact = main == sorted(g["det_idx"])
        alt_better = any(score_positions(a, g["det_idx"]).f1 > f_main for a in alts)
        just = " ".join(str(mi.get(k) or "") for k in ("justification", "validation")).strip()
        foreign = sorted({c for c in SIGN_CODE.findall(just) if c not in set(g["tokens"])})
        raw = parse_answer((rec.get("raw_answers") or {}).get("synthesizer", {}).get("content") or "")
        raw_main = _interp_idx(raw.get("main_interpretation") or {}, legacy) if raw else []
        amb = sj.get("ambiguity") or {}
        amb_text = "\n".join(f"- {x}" for x in (amb.get("factors") or []))
        if amb.get("consensus"):
            amb_text += f"\nConsenso: {amb['consensus']}"
        rec_text = "\n".join(f"- {x}" for x in (sj.get("recommendation") or []))
        rows.append({
            "id": rec["id"], "mdc": rec["mdc"], "translation": g.get("translation", ""),
            "transliteration": g.get("transliteration", ""),
            "seq_gold": _indexed_sequence(legacy, g["det_idx"]),
            "gold": _legacy_label(legacy, g["det_idx"]),
            "pred": _legacy_label(legacy, main),
            "raw_pred": _legacy_label(legacy, raw_main) if raw else "(no disponible)",
            "justification": just or "(sin justificación)",
            "alternatives": _alts_text(sj, legacy),
            "ambiguity": (amb_text + ("\nRecomendaciones:\n" + rec_text if rec_text else "")).strip() or "(vacío)",
            "foreign": ", ".join(foreign) or "(ninguno)",
            "main_exact": main_exact, "alt_better": alt_better, "has_foreign": bool(foreign),
        })
    return rows


def sample(rows: List[dict], seed: int) -> List[dict]:
    rng = random.Random(seed)
    pools = {
        "D_codigos_ajenos": [r for r in rows if r["has_foreign"]],
        "B_alternativa_mejor": [r for r in rows if not r["main_exact"] and r["alt_better"]],
        "C_sin_alternativa_mejor": [r for r in rows if not r["main_exact"] and not r["alt_better"]],
        "A_principal_correcta": [r for r in rows if r["main_exact"]],
    }
    chosen, used = [], set()
    for name in ("D_codigos_ajenos", "B_alternativa_mejor", "C_sin_alternativa_mejor", "A_principal_correcta"):
        want = dict(STRATA)[name]
        pool = [r for r in pools[name] if r["id"] not in used]
        rng.shuffle(pool)
        for r in pool[:want]:
            chosen.append(dict(r, stratum=name))
            used.add(r["id"])
    target = sum(n for _, n in STRATA)
    for name in ("C_sin_alternativa_mejor", "A_principal_correcta"):   # top up if a stratum was short
        pool = [r for r in pools[name] if r["id"] not in used]
        rng.shuffle(pool)
        for r in pool[:max(0, target - len(chosen))]:
            chosen.append(dict(r, stratum=name))
            used.add(r["id"])
    rng.shuffle(chosen)
    return chosen


# ── workbook ────────────────────────────────────────────────────────────────

COLUMNS = [  # (header, key or None for input, width)
    ("Nº", "n", 5), ("ID", "id", 13), ("Secuencia MdC", "mdc", 34), ("Traducción (corpus)", "translation", 34),
    ("Secuencia indexada; [ ] = determinativo de referencia", "seq_gold", 46),
    ("Determinativos de referencia", "gold", 24), ("Predicción publicada", "pred", 24),
    ("Predicción propia del sintetizador (antes del posprocesamiento)", "raw_pred", 24),
    ("Justificación publicada", "justification", 60), ("Alternativas publicadas y condición de promoción", "alternatives", 60),
    ("Ambigüedad y recomendaciones", "ambiguity", 46), ("Códigos citados ausentes de la secuencia (automático)", "foreign", 18),
    ("C1 Corrección egiptológica de la justificación (1-3)", None, 14),
    ("C2 La justificación corresponde a la predicción publicada (1-3)", None, 14),
    ("C3 Utilidad de las alternativas y condiciones (1-3 / N/A)", None, 14),
    ("C4 Códigos ausentes: error o comparación legítima", None, 16),
    ("Comentarios", None, 40),
]
INPUT_COLS = [i + 1 for i, (_, k, _) in enumerate(COLUMNS) if k is None]


def _font(bold=False, size=10, color="000000", italic=False):
    return Font(name=FONT, bold=bold, size=size, color=color, italic=italic)


def write_workbook(path: str, chosen: List[dict]) -> None:
    wb = Workbook()
    ins = wb.active
    ins.title = "Instrucciones"
    lines = [
        ("Evaluación experta de justificaciones y alternativas", True, 14),
        ("Proyecto: detección de determinativos en secuencias jeroglíficas (MdC). Sistema evaluado: versión del manuscrito (v1.47), modelo gpt-5.4-nano.", False, 10),
        ("", False, 10),
        ("Qué hacer", True, 11),
        ("1. Escriba su nombre en la celda B12 de esta hoja.", False, 10),
        (f"2. En la hoja «Evaluación», complete solo las celdas amarillas (columnas M a Q) de las {len(chosen)} filas.", False, 10),
        ("3. Evalúe cada fila de forma independiente; no hace falta seguir el orden. No consulte la evaluación del otro evaluador.", False, 10),
        ("4. Las posiciones se escriben signo@índice; el índice cuenta también los separadores (-, :, *), igual que en los textos del sistema.", False, 10),
        ("5. Las justificaciones citan reglas internas del sistema («Regla 1» a «Regla 5») y agentes (1 = semántico, 2 = estructural, 3 = reglas). No evalúe esas etiquetas; evalúe el razonamiento egiptológico.", False, 10),
        ("6. La hoja «Resumen» se calcula sola.", False, 10),
        ("", False, 10),
    ]
    for r, (txt, bold, size) in enumerate(lines, 1):
        c = ins.cell(row=r, column=1, value=txt)
        c.font = _font(bold=bold, size=size)
    ins.cell(row=12, column=1, value="Evaluador:").font = _font(bold=True)
    ev = ins.cell(row=12, column=2)
    ev.fill, ev.border = FILL_INPUT, BORDER
    ins.cell(row=13, column=1, value="Fecha:").font = _font(bold=True)
    d = ins.cell(row=13, column=2)
    d.fill, d.border = FILL_INPUT, BORDER

    crit = [
        ("Criterios", None),
        ("C1 Corrección egiptológica", "3 = correcta y pertinente; 2 = parcialmente correcta o con imprecisiones; 1 = incorrecta o irrelevante."),
        ("C2 Correspondencia", "3 = explica exactamente los signos de la predicción publicada; 2 = explica parte; 1 = explica otra predicción o no explica."),
        ("C3 Alternativas", "3 = al menos una alternativa es plausible y su condición de promoción es verificable; 2 = plausible pero condición vaga; 1 = no aportan; N/A = no hay alternativas."),
        ("C4 Códigos ausentes", "Solo si la columna L lista códigos: «Error» (alucinación), «Comparación legítima» (se citan a propósito) o «Mixto». Si L dice (ninguno): N/A."),
    ]
    r0 = 15
    for k, (a, b) in enumerate(crit):
        ca = ins.cell(row=r0 + k, column=1, value=a)
        ca.font = _font(bold=True if b is None or k == 0 else True)
        if b:
            cb = ins.cell(row=r0 + k, column=2, value=b)
            cb.font = _font()
            cb.alignment = Alignment(wrap_text=True, vertical="top")

    ex0 = r0 + len(crit) + 1
    ins.cell(row=ex0, column=1, value="Ejemplo de respuesta (formato esperado; no es una fila real)").font = _font(bold=True)
    ex_headers = ["C1", "C2", "C3", "C4", "Comentarios"]
    ex_vals = [2, 1, 3, "Comparación legítima",
               "La justificación explica A1 y Z1, pero la predicción publicada también incluye D54, que no se menciona."]
    for j, (h, v) in enumerate(zip(ex_headers, ex_vals)):
        ins.cell(row=ex0 + 1, column=1 + j, value=h).font = _font(bold=True)
        c = ins.cell(row=ex0 + 2, column=1 + j, value=v)
        c.font = _font(italic=True, color="595959")
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ins.column_dimensions["A"].width = 34
    ins.column_dimensions["B"].width = 95
    for col in "CDE":
        ins.column_dimensions[col].width = 22

    ws = wb.create_sheet("Evaluación")
    for j, (h, _, w) in enumerate(COLUMNS, 1):
        c = ws.cell(row=1, column=j, value=h)
        c.font = _font(bold=True)
        c.fill = FILL_HEAD
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        c.border = BORDER
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.row_dimensions[1].height = 75
    for i, row in enumerate(chosen, 2):
        row = dict(row, n=i - 1)
        for j, (_, key, _) in enumerate(COLUMNS, 1):
            c = ws.cell(row=i, column=j, value=row[key] if key else None)
            c.font = _font()
            c.alignment = Alignment(wrap_text=True, vertical="top")
            c.border = BORDER
            if key is None:
                c.fill = FILL_INPUT
        lines_needed = max(
            sum(len(part) // max(1, int(w * 1.15)) + 1 for part in str(row[key] or "").split("\n"))
            for _, key, w in COLUMNS if key)
        ws.row_dimensions[i].height = min(400, max(45, 13 * lines_needed))
    last = len(chosen) + 1
    ws.freeze_panes = "C2"
    ws.cell(row=1, column=12).comment = Comment(
        "Códigos de signos citados en la justificación que no aparecen en la secuencia (detección automática).", "egydet")

    def dv(formula, cols, prompt):
        v = DataValidation(type="list", formula1=formula, allow_blank=True, showErrorMessage=True,
                           errorTitle="Valor no válido", error=prompt)
        ws.add_data_validation(v)
        for col in cols:
            v.add(f"{get_column_letter(col)}2:{get_column_letter(col)}{last}")

    dv('"1,2,3"', [13, 14], "Use 1, 2 o 3.")
    dv('"1,2,3,N/A"', [15], "Use 1, 2, 3 o N/A.")
    dv('"Error,Comparación legítima,Mixto,N/A"', [16], "Elija una opción de la lista.")

    sm = wb.create_sheet("Resumen")
    sm["A1"] = "Resumen (se calcula automáticamente a partir de la hoja «Evaluación»)"
    sm["A1"].font = _font(bold=True, size=12)
    heads = ["Criterio", "Media", "Nº evaluadas", "Nº = 1", "Nº = 2", "Nº = 3", "Nº N/A"]
    for j, h in enumerate(heads, 1):
        c = sm.cell(row=3, column=j, value=h)
        c.font, c.fill, c.border = _font(bold=True), FILL_HEAD, BORDER
    for k, (label, col) in enumerate((("C1 Corrección egiptológica", "M"), ("C2 Correspondencia", "N"),
                                      ("C3 Alternativas", "O")), 4):
        rng = f"'Evaluación'!{col}2:{col}{last}"
        vals = [label, f"=IFERROR(AVERAGE({rng}),\"-\")", f"=COUNT({rng})",
                f"=COUNTIF({rng},1)", f"=COUNTIF({rng},2)", f"=COUNTIF({rng},3)", f"=COUNTIF({rng},\"N/A\")"]
        for j, v in enumerate(vals, 1):
            c = sm.cell(row=k, column=j, value=v)
            c.font, c.border = _font(), BORDER
            if j == 2:
                c.number_format = "0.00"
    sm.cell(row=8, column=1, value="C4 Códigos ausentes").font = _font(bold=True)
    for k, opt in enumerate(("Error", "Comparación legítima", "Mixto", "N/A"), 9):
        sm.cell(row=k, column=1, value=opt).font = _font()
        c = sm.cell(row=k, column=2, value=f"=COUNTIF('Evaluación'!P2:P{last},\"{opt}\")")
        c.font, c.border = _font(), BORDER
    sm.cell(row=14, column=1, value="Filas evaluadas (C1 completo)").font = _font(bold=True)
    sm.cell(row=14, column=2, value=f"=COUNT('Evaluación'!M2:M{last})").font = _font()
    sm.cell(row=15, column=1, value="Filas en la muestra").font = _font(bold=True)
    sm.cell(row=15, column=2, value=f"=COUNTA('Evaluación'!B2:B{last})").font = _font()
    sm.column_dimensions["A"].width = 32
    for col in "BCDEFG":
        sm.column_dimensions[col].width = 14

    wb.active = 1
    wb.save(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build the expert evaluation workbook")
    ap.add_argument("run_dir")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    chosen = sample(build_rows(args.run_dir), args.seed)
    xlsx = os.path.join(args.run_dir, "evaluacion_expertos.xlsx")
    write_workbook(xlsx, chosen)
    with open(os.path.join(args.run_dir, "evaluacion_clave.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "id", "stratum", "main_exact", "alt_better", "has_foreign"])
        for n, r in enumerate(chosen, 1):
            w.writerow([n, r["id"], r["stratum"], r["main_exact"], r["alt_better"], r["has_foreign"]])
    print(f"{len(chosen)} rows -> {xlsx}")


if __name__ == "__main__":
    main()
