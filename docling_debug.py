"""
Docling – inšpektor surových atribútov výstupu.
Spustenie: python inspect_docling.py
"""

import os
import random
import sys
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built     = lambda: False

from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, AcceleratorOptions,
)

PNG_FOLDER = Path("data/DocLayNet/PNG")


def section(title: str):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def safe_attrs(obj) -> dict:
    """Vráti atribúty objektu (funguje pre bežné triedy aj Pydantic v1/v2)."""
    if obj is None:
        return {}
    if hasattr(obj, "__dict__") and obj.__dict__:
        return dict(obj.__dict__)
    if hasattr(obj, "model_fields"):          # pydantic v2
        return {k: getattr(obj, k, "?") for k in obj.model_fields}
    if hasattr(obj, "__fields__"):            # pydantic v1
        return {k: getattr(obj, k, "?") for k in obj.__fields__}
    return {}


def dump_attrs(obj, label="obj", indent=2, max_depth=3, _depth=0):
    pad = " " * indent * _depth
    attrs = safe_attrs(obj)
    if not attrs:
        print(f"{pad}  (žiadne atribúty alebo primitív: {obj!r})")
        return
    for name, val in attrs.items():
        typ = type(val).__name__
        if isinstance(val, (int, float, str, bool, type(None))):
            print(f"{pad}├── {name}: {typ} = {val!r}")
        elif isinstance(val, (list, tuple)):
            print(f"{pad}├── {name}: {typ}[{len(val)}]", end="")
            if val and _depth < max_depth:
                print(f"  ← prvý prvok ({type(val[0]).__name__}):")
                dump_attrs(val[0], name, indent, max_depth, _depth + 1)
            else:
                print()
        elif _depth < max_depth:
            print(f"{pad}├── {name}: {typ}")
            dump_attrs(val, name, indent, max_depth, _depth + 1)
        else:
            print(f"{pad}├── {name}: {typ}  ...")


def find_key(obj, target: str, path="root", _visited=None, _found=None) -> list:
    if _visited is None:
        _visited = set()
    if _found is None:
        _found = []
    oid = id(obj)
    if oid in _visited:
        return _found
    _visited.add(oid)

    attrs = safe_attrs(obj)
    for name, val in attrs.items():
        full = f"{path}.{name}"
        if target.lower() in name.lower():
            _found.append(f"{full}  →  {type(val).__name__} = {val!r}")
        if isinstance(val, (list, tuple)) and val:
            find_key(val[0], target, f"{full}[0]", _visited, _found)
        elif not isinstance(val, (int, float, str, bool, type(None))):
            find_key(val, target, full, _visited, _found)
    return _found


def main():
    if not PNG_FOLDER.exists():
        print(f"❌ Priečinok neexistuje: {PNG_FOLDER}")
        sys.exit(1)

    png_files = list(PNG_FOLDER.glob("*.png"))
    if not png_files:
        print(f"❌ Žiadne PNG v {PNG_FOLDER}")
        sys.exit(1)

    random_img = random.choice(png_files)
    print(f"🎲 Obrázok: {random_img.name}")

    # PdfPipelineOptions funguje aj pre obrázky v tejto verzii Docling
    opts = PdfPipelineOptions()
    opts.do_ocr = False

    converter = DocumentConverter(
        format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)}
    )
    print("⏳ Spracovávam...")
    result = converter.convert(str(random_img))
    print("✅ Hotovo.\n")

    # ── -1. Čo vôbec exportuje pipeline_options modul ───
    section("-1. DOSTUPNÉ TRIEDY V pipeline_options")
    import docling.datamodel.pipeline_options as _po
    exported = [x for x in dir(_po) if not x.startswith("_")]
    for name in exported:
        obj = getattr(_po, name)
        print(f"  {name:40} {type(obj).__name__}")

    # ── -1b. Verzia Docling ───────────────────────────────
    section("-1b. VERZIA DOCLING")
    try:
        import importlib.metadata
        print(f"  docling verzia: {importlib.metadata.version('docling')}")
    except Exception as e:
        print(f"  {e}")


    section("0. TYP A OBSAH result.pages")
    print(f"  type(result.pages)  = {type(result.pages)}")
    print(f"  type(result)        = {type(result)}")
    print(f"  result attrs        = {list(safe_attrs(result).keys())}")

    if isinstance(result.pages, dict):
        print(f"  Kľúče (page_no):    {list(result.pages.keys())}")
        page = next(iter(result.pages.values()))
    elif isinstance(result.pages, list):
        print(f"  Počet stránok:      {len(result.pages)}")
        page = result.pages[0]
    else:
        print(f"  Neznámy typ pages: {type(result.pages)}")
        sys.exit(1)

    print(f"\n  type(page)          = {type(page)}")
    print(f"  page attrs          = {list(safe_attrs(page).keys())}")

    # ── 1. predictions ────────────────────────────────────
    section("1. ŠTRUKTÚRA page.predictions")
    predictions = getattr(page, "predictions", None)
    print(f"  type(predictions)   = {type(predictions)}")
    print(f"  predictions attrs   = {list(safe_attrs(predictions).keys())}")

    layout = getattr(predictions, "layout", None)
    print(f"\n  type(layout)        = {type(layout)}")
    if layout is not None:
        layout_attrs = safe_attrs(layout)
        print(f"  layout attrs        = {list(layout_attrs.keys())}")
        clusters = getattr(layout, "clusters", None)
        print(f"  clusters type       = {type(clusters)}")
        print(f"  clusters count      = {len(clusters) if clusters else 0}")
    else:
        print("  ⚠️  layout je None – hľadáme inde...")
        dump_attrs(predictions, max_depth=3)

    # ── 2. Fallback – hľadaj clustery kdekoľvek ──────────
    section("2. HĽADANIE 'cluster' KDEKOĽVEK V RESULT")
    hits = find_key(result, "cluster")
    for h in hits[:20]:
        print(f"  {h}")
    if not hits:
        print("  (nič nenájdené)")

    # ── 3. Hľadanie score / confidence ───────────────────
    section("3. HĽADANIE 'score' A 'confidence' V RESULT")
    for kw in ["score", "confidence", "prob"]:
        hits = find_key(result, kw)
        if hits:
            print(f"\n  ── '{kw}' ──")
            for h in hits[:15]:
                print(f"    {h}")
        else:
            print(f"\n  '{kw}': nenájdené")

    # ── 4. Surový strom result (hĺbka 2) ─────────────────
    section("4. CELÝ STROM result (hĺbka 2)")
    dump_attrs(result, max_depth=2)

    # ── 5. Ak sú clustery, ukáž ich ──────────────────────
    if layout and clusters:
        section(f"5. PRVÝCH 5 CLUSTEROV (z {len(clusters)})")
        for i, c in enumerate(clusters[:5]):
            print(f"\n  ── Cluster [{i+1}] ──")
            for k, v in safe_attrs(c).items():
                print(f"    {k:25} = {v!r}")

        section("6. ZÁVER – kandidáti na skóre")
        cl = clusters[0]
        cl_attrs = list(safe_attrs(cl).keys())
        print(f"  Atribúty clustra: {cl_attrs}")
        candidates = [a for a in cl_attrs if any(
            kw in a.lower() for kw in ["score", "conf", "prob", "logit"]
        )]
        if candidates:
            print(f"\n  ✅ Nájdené: {candidates}")
            for sc in candidates:
                vals = [getattr(c, sc, None) for c in clusters if getattr(c, sc, None) is not None]
                if vals:
                    print(f"     {sc}: min={min(vals):.4f}  max={max(vals):.4f}  príklady={vals[:3]}")
        else:
            print("\n  ⚠️  Žiadne score/confidence atribúty na cluster objekte.")
    else:
        section("5. CLUSTERS – ŽIADNE (ďalší debug)")
        print("  Celý strom page (hĺbka 4):")
        dump_attrs(page, max_depth=4)


if __name__ == "__main__":
    main()