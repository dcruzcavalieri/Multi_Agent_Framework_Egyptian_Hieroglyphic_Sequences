"""Parse the sign -> phonetic value list (Appendix B) of the online summary of the Manuel de Codage.

Source: H. van den Berg, "Manuel de Codage. A standard system for the computer-encoding of Egyptian
transliteration and hieroglyphic texts", Centre for Computer-aided Egyptological Research (CCER), 1997,
page saved as HTML; the list goes back to Buurman et al., Informatique et Egyptologie 2 (Paris 1988).
The page carries a copyright notice, so the parsed table is kept in kb/private/ (git-ignored).

    python tools/parse_mdc_phonetic.py "kb/private/mdc_page/Manuel de Codage.html" kb/private/mdc_phonetic_values.json

Values are MdC ASCII transliteration (A = aleph, i = yod, a = ayin, H, x, X, S, T, D ...). In the page the
Aa signs are written with the old prefix J (J1 = Aa1); they are renamed here.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict


def parse(path: str) -> dict:
    raw = open(path, encoding="utf-8", errors="replace").read()
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</tr>|</h\d>|</li>", "\n", raw)
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    start = text.rfind("Appendix B: List of sign numbers and phonetic values")
    end = text.rfind("Appendix C: List of phonetic values and sign numbers")
    if start < 0 or end < start:
        raise ValueError("Appendix B not found")
    values = defaultdict(list)
    for code, val in re.findall(r"\b([A-Z]{1,2}\d+[A-Za-z]?)\s*=\s*(\S+)", text[start:end]):
        code = re.sub(r"^J(?=\d)", "Aa", code)
        if val not in values[code]:
            values[code].append(val)
    return dict(sorted(values.items()))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("html")
    ap.add_argument("out_json")
    args = ap.parse_args(argv)
    values = parse(args.html)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump({"source": "Manuel de Codage, CCER online summary (1997), Appendix B",
                   "values": values}, fh, indent=1, ensure_ascii=False)
    print(f"{sum(len(v) for v in values.values())} values for {len(values)} signs -> {args.out_json}")


if __name__ == "__main__":
    main()
