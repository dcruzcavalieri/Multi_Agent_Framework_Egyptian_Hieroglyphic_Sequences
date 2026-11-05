"""Extract sign -> transliteration values from JSesh's sign description file.

Source: S. Rosmorduc, JSesh, file ``jsesh/src/main/resources/jsesh/glyphs/resources/signs_description.xml``
(licensed CC-BY 4.0 according to the JSesh README; the rest of JSesh is CeCILL-C). The output can
therefore be redistributed with attribution; the attribution is written into the JSON.

    python tools/parse_jsesh_signs.py <jsesh-master.zip or signs_description.xml> kb/jsesh_sign_values.json

For each sign: its transliterations (MdC ASCII: A aleph, i yod, a ayin, H, x, X, S, T, D ...) with their
type (phonogram, ideogram, abbreviation, typical), and its base sign when JSesh records it as a variant.
Variants without values of their own inherit those of a base sign that is a full linguistic variant.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
import xml.etree.ElementTree as ET

INNER = "jsesh/src/main/resources/jsesh/glyphs/resources/signs_description.xml"


def _read(path: str) -> bytes:
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(INNER))
            return z.read(name)
    with open(path, "rb") as fh:
        return fh.read()


def parse(path: str) -> dict:
    data = _read(path)
    # the DOCTYPE points to a DTD that is not needed for parsing
    root = ET.parse(io.BytesIO(data), parser=ET.XMLParser()).getroot()
    values, types, base = {}, {}, {}
    for s in root.iter("sign"):
        code = s.get("sign")
        for t in s.iter("hasTransliteration"):
            v, ty = t.get("transliteration"), t.get("type") or ""
            if v and v not in values.setdefault(code, []):
                values[code].append(v)
                types.setdefault(code, []).append(ty)
        for var in s.iter("variantOf"):
            if var.get("linguistic") == "full" and var.get("baseSign"):
                base.setdefault(code, var.get("baseSign"))
    inherited = 0
    for code, b in base.items():
        if code not in values and b in values:
            values[code], types[code] = list(values[b]), list(types[b])
            inherited += 1
    return {"source": "JSesh (S. Rosmorduc), signs_description.xml, CC-BY 4.0",
            "values": dict(sorted(values.items())), "types": dict(sorted(types.items())),
            "variant_of": dict(sorted(base.items())), "n_inherited": inherited}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("out_json")
    args = ap.parse_args(argv)
    res = parse(args.src)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    n = sum(len(v) for v in res["values"].values())
    print(f"{n} values for {len(res['values'])} signs ({res['n_inherited']} inherited from base signs) -> {args.out_json}")


if __name__ == "__main__":
    main()
