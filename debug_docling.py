

import os
import torch

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built     = lambda: False

# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════
PNG_FOLDER = "data/DocLayNet/PNG"
OUT_JSON   = "data/output_json2"
OUT_VIS    = "data/output_visual2"
MAX_IMAGES = 100          # None = všetky
# ══════════════════════════════════════════════════════════

import json, sys, base64
from pathlib import Path
from collections import defaultdict
from PIL import Image as PILImage

LABEL_COLORS = {
    "title":               "#e63946",
    "section_header":      "#f4a261",
    "text":                "#2a9d8f",
    "paragraph":           "#2a9d8f",
    "caption":             "#457b9d",
    "table":               "#8338ec",
    "figure":              "#06d6a0",
    "list_item":           "#ffb703",
    "page_header":         "#adb5bd",
    "page_footer":         "#adb5bd",
    "footnote":            "#6c757d",
    "formula":             "#e9c46a",
    "code":                "#264653",
    "picture":             "#06d6a0",
    "checkbox_selected":   "#fb8500",
    "checkbox_unselected": "#fb8500",
}
DEFAULT_COLOR = "#999999"


# ════════════════════════════════════════════════════════
#  1. DOCLING
# ════════════════════════════════════════════════════════

def run_dla_on_image(img_path: Path):
    from docling.document_converter import DocumentConverter, ImageFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions, EasyOcrOptions, AcceleratorOptions
    )

    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.ocr_options = EasyOcrOptions(lang=["sk", "en"])
    opts.do_table_structure = False
    opts.images_scale = 2.0
    opts.accelerator_options = AcceleratorOptions(device="cpu", num_threads=4)

    converter = DocumentConverter(
        format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)}
    )
    return converter.convert(str(img_path))




# ════════════════════════════════════════════════════════
#  3. EXTRAKCIA  →  YOLO-style nodes
# ════════════════════════════════════════════════════════

def extract_nodes(result) -> tuple[list[dict], float, float]:
    """
    Vráti (nodes, page_width, page_height).

    Postup:
      1. Načítame raw clustery z result.pages[0].predictions.layout.clusters
         → label, bbox (TOPLEFT), confidence
      2. Spárujeme text z result.assembled.elements cez cluster.id
      3. IoA filtering
      4. Sformátujeme ako YOLO-style node
    """
    # ── Stránka ──────────────────────────────────────────
    page = result.pages[0]
    pw = page.size.width  if page.size else 1.0
    ph = page.size.height if page.size else 1.0

    # ── Text lookup: cluster_id → text ───────────────────
    text_by_cluster_id: dict[int, str] = {}
    assembled = getattr(page, "assembled", None)
    if assembled:
        for el in getattr(assembled, "elements", []):
            cl = getattr(el, "cluster", None)
            if cl is not None:
                cid  = getattr(cl, "id", None)
                text = getattr(el, "text", "") or ""
                if cid is not None:
                    text_by_cluster_id[cid] = text.strip()

    # ── Raw clustery ─────────────────────────────────────
    layout = getattr(getattr(page, "predictions", None), "layout", None)
    if not layout or not getattr(layout, "clusters", None):
        return [], pw, ph

    raw = []
    for cl in layout.clusters:
        bbox  = cl.bbox           # TOPLEFT origin
        label = cl.label
        label_str = label.value if hasattr(label, "value") else str(label)
        conf  = float(getattr(cl, "confidence", 0.0))
        cid   = getattr(cl, "id", None)

        raw.append({
            "cluster_id": cid,
            "label":      label_str,
            "confidence": conf,
            "text":       text_by_cluster_id.get(cid, ""),
            # absolútne px súradnice (TOPLEFT) – rovnaký formát ako YOLO
            "coords":     [bbox.l, bbox.t, bbox.r, bbox.b],
        })

    # ── IoA filter ───────────────────────────────────────
    filtered = raw

    # ── Zostavenie finálnych nodov ────────────────────────
    # Unikátne labely → integer id (abecedne)
    all_labels  = sorted({n["label"] for n in filtered})
    label_to_id = {l: i for i, l in enumerate(all_labels)}

    nodes = []
    for node_id, n in enumerate(filtered):
        x1, y1, x2, y2 = n["coords"]

        norm_x1 = x1 / pw
        norm_y1 = y1 / ph
        norm_x2 = x2 / pw
        norm_y2 = y2 / ph

        nodes.append({
            "node_id":    node_id,
            "label":      n["label"],
            "label_id":   label_to_id[n["label"]],
            "confidence": round(n["confidence"], 6),
            "text":       n["text"],
            "geometry": {
                "absolute_pixel_coords": [int(x1), int(y1), int(x2), int(y2)],
                "normalized_coords":     [
                    round(norm_x1, 6), round(norm_y1, 6),
                    round(norm_x2, 6), round(norm_y2, 6),
                ],
                "normalized_center": [
                    round((norm_x1 + norm_x2) / 2, 6),
                    round((norm_y1 + norm_y2) / 2, 6),
                ],
                "normalized_size": [
                    round(norm_x2 - norm_x1, 6),
                    round(norm_y2 - norm_y1, 6),
                ],
            },
        })

    return nodes, pw, ph


# ════════════════════════════════════════════════════════
#  4. JSON EXPORT
# ════════════════════════════════════════════════════════

def save_json(nodes, img_path: Path, pw: float, ph: float, out_dir: Path) -> Path:
    data = {
        "file_name":   img_path.name,
        "page_width":  pw,
        "page_height": ph,
        "node_count":  len(nodes),
        "docling_nodes": nodes,
    }
    out_path = out_dir / f"{img_path.stem}.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return out_path


# ════════════════════════════════════════════════════════
#  5. DIAGNOSTICS
# ════════════════════════════════════════════════════════

def build_diagnostics(all_results: list[dict], out_dir: Path) -> Path:
    label_counts = defaultdict(int)
    label_confs  = defaultdict(list)
    total = 0

    for r in all_results:
        for n in r["nodes"]:
            lbl = n["label"]
            label_counts[lbl] += 1
            total += 1
            label_confs[lbl].append(n["confidence"])

    lines = ["═" * 62, "  DOCLING DLA – DIAGNOSTICS", "═" * 62, ""]

    lines += ["── OCR Engine ─────────────────────────────────────",
              "   EasyOCR (sk, en) → CPU (layout model) + CPU (OCR)", ""]

    lines += ["── Spracované obrázky ─────────────────────────────"]
    for r in all_results:
        lines.append(f"   {r['img_name']:<60}  {len(r['nodes']):>3} nodov")
    lines.append("")

    lines += ["── Distribúcia tried ──────────────────────────────",
              f"   {'Trieda':<28} {'N':>5}  {'%':>6}", "   " + "-" * 44]
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / total if total else 0
        lines.append(f"   {lbl:<28} {cnt:>5}  {pct:>5.1f}%")
    lines.append(f"   {'SPOLU':<28} {total:>5}")
    lines.append("")

    lines += ["── Confidence per trieda ──────────────────────────",
              f"   {'Trieda':<28} {'N':>4}  {'Min':>6}  {'Avg':>6}  {'Max':>6}  Distribúcia",
              "   " + "-" * 70]
    for lbl, scores in sorted(label_confs.items()):
        n   = len(scores)
        mn  = min(scores)
        avg = sum(scores) / n
        mx  = max(scores)
        bar = "█" * int(avg * 24) + "░" * (24 - int(avg * 24))
        lines.append(f"   {lbl:<28} {n:>4}  {mn:>6.3f}  {avg:>6.3f}  {mx:>6.3f}  |{bar}|")

    lines += ["", "═" * 62]
    out = out_dir / "diagnostics.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ════════════════════════════════════════════════════════
#  6. HTML VIEWER
# ════════════════════════════════════════════════════════

def img_to_b64(img_path: Path) -> str:
    return base64.b64encode(img_path.read_bytes()).decode()


def nodes_to_js(nodes, img_w, img_h, pw, ph) -> str:
    items = []
    for n in nodes:
        lbl   = n["label"]
        color = LABEL_COLORS.get(lbl, DEFAULT_COLOR)
        text  = n["text"].replace("`", "'").replace("\\", "\\\\").replace("\n", " ")
        conf  = f"{n['confidence']:.3f}"
        # bbox je TOPLEFT, obrázok tiež TOPLEFT → priamy prevod
        sx = img_w / pw if pw else 1
        sy = img_h / ph if ph else 1
        x1, y1, x2, y2 = n["geometry"]["absolute_pixel_coords"]
        x = x1 * sx; y = y1 * sy
        w = (x2 - x1) * sx; h = (y2 - y1) * sy
        items.append(
            f'{{label:"{lbl}",x:{x:.1f},y:{y:.1f},w:{w:.1f},h:{h:.1f},'
            f'color:"{color}",text:`{text}`,conf:"{conf}"}}'
        )
    return "[" + ",\n".join(items) + "]"


def build_html(page_data: list[dict], out_dir: Path) -> Path:
    pages_js_parts = []
    for pd in page_data:
        overlay   = nodes_to_js(pd["nodes"], pd["img_w"], pd["img_h"], pd["pw"], pd["ph"])
        seen_lbls = sorted({n["label"] for n in pd["nodes"]})
        legend_js = json.dumps(
            [{"label": l, "color": LABEL_COLORS.get(l, DEFAULT_COLOR)} for l in seen_lbls]
        )
        pages_js_parts.append(
            f'{{name:{json.dumps(pd["img_name"])},'
            f'b64:{json.dumps("data:image/png;base64," + pd["b64"])},'
            f'w:{pd["img_w"]},h:{pd["img_h"]},'
            f'elements:{overlay},legend:{legend_js}}}'
        )

    pages_js = "[\n" + ",\n".join(pages_js_parts) + "\n]"

    html = f"""<!DOCTYPE html>
<html lang="sk">
<head>
<meta charset="UTF-8">
<title>DLA Viewer</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0d0f14;--panel:#161922;--border:#252a35;
  --accent:#4f8ef7;--text:#c8cdd8;--muted:#555e70;
  --radius:8px;--font:'IBM Plex Mono','Fira Code','Courier New',monospace
}}
body{{background:var(--bg);color:var(--text);font-family:var(--font);
  font-size:13px;height:100vh;display:flex;flex-direction:column;overflow:hidden}}
header{{background:var(--panel);border-bottom:1px solid var(--border);
  padding:10px 20px;display:flex;align-items:center;gap:16px;flex-shrink:0}}
header h1{{font-size:14px;font-weight:600;color:var(--accent);
  letter-spacing:.05em;white-space:nowrap}}
.pi{{color:var(--muted);font-size:11px;flex:1}}
.nb{{background:var(--border);color:var(--text);border:1px solid var(--border);
  padding:5px 14px;border-radius:var(--radius);cursor:pointer;
  font-family:var(--font);font-size:12px;transition:background .15s,border-color .15s}}
.nb:hover{{background:var(--accent);border-color:var(--accent);color:#fff}}
.nb:disabled{{opacity:.3;cursor:default}}
.pc{{color:var(--muted);font-size:11px;min-width:80px;text-align:center}}
#ps{{background:var(--border);color:var(--text);border:1px solid var(--border);
  padding:4px 8px;border-radius:var(--radius);font-family:var(--font);
  font-size:11px;cursor:pointer;max-width:240px}}
main{{display:flex;flex:1;overflow:hidden}}
#cw{{flex:1;overflow:auto;display:flex;align-items:flex-start;
  justify-content:center;padding:20px;position:relative}}
#stage{{position:relative;display:inline-block;flex-shrink:0}}
#pi{{display:block;max-width:100%;height:auto;border-radius:var(--radius);
  box-shadow:0 8px 40px rgba(0,0,0,.6)}}
#oc{{position:absolute;top:0;left:0;pointer-events:none}}
#tt{{position:fixed;background:#1a1f2e;border:1px solid var(--accent);
  border-radius:var(--radius);padding:10px 14px;max-width:400px;font-size:11px;
  line-height:1.6;pointer-events:none;z-index:9999;display:none;
  box-shadow:0 4px 24px rgba(0,0,0,.6)}}
.ttl{{font-weight:700;font-size:10px;letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px}}
.ttt{{color:#e0e4ef;word-break:break-word;white-space:pre-wrap}}
.ttc{{margin-top:6px;color:var(--muted);font-size:10px}}
#sb{{width:230px;flex-shrink:0;background:var(--panel);
  border-left:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden}}
#sb h2{{font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);
  padding:14px 16px 8px;border-bottom:1px solid var(--border)}}
#leg{{overflow-y:auto;flex:1;padding:8px 0}}
.li{{display:flex;align-items:center;gap:10px;padding:5px 16px;cursor:pointer;
  transition:background .1s;border-radius:4px;margin:1px 6px}}
.li:hover{{background:var(--border)}}
.li.muted{{opacity:.3}}
.ld{{width:12px;height:12px;border-radius:3px;flex-shrink:0}}
.ll{{font-size:11px;flex:1}}
.lc{{font-size:10px;color:var(--muted)}}
#stb{{border-top:1px solid var(--border);padding:12px 16px;
  font-size:10px;color:var(--muted);line-height:1.8}}
</style>
</head>
<body>
<header>
  <h1>◈ DLA Viewer</h1>
  <select id="ps"></select>
  <span class="pi" id="pi2"></span>
  <button class="nb" id="bP">◀ Prev</button>
  <span class="pc" id="pc"></span>
  <button class="nb" id="bN">Next ▶</button>
</header>
<main>
  <div id="cw">
    <div id="stage">
      <img id="pi" src="" alt="">
      <canvas id="oc"></canvas>
    </div>
  </div>
  <div id="sb">
    <h2>Triedy</h2>
    <div id="leg"></div>
    <div id="stb"></div>
  </div>
</main>
<div id="tt">
  <div class="ttl" id="ttl"></div>
  <div class="ttt" id="ttt"></div>
  <div class="ttc" id="ttc"></div>
</div>
<script>
const PAGES={pages_js};
let cur=0,hidden=new Set(),hov=null;
const img=document.getElementById("pi"),
      cv=document.getElementById("oc"),ctx=cv.getContext("2d"),
      tt=document.getElementById("tt"),
      ttl=document.getElementById("ttl"),ttt=document.getElementById("ttt"),
      ttc=document.getElementById("ttc"),
      leg=document.getElementById("leg"),stb=document.getElementById("stb"),
      pi2=document.getElementById("pi2"),pc=document.getElementById("pc"),
      bP=document.getElementById("bP"),bN=document.getElementById("bN"),
      ps=document.getElementById("ps");

PAGES.forEach((p,i)=>{{
  const o=document.createElement("option");
  o.value=i;o.textContent=p.name;ps.appendChild(o);
}});

function rgb(h){{
  return `${{parseInt(h.slice(1,3),16)}},${{parseInt(h.slice(3,5),16)}},${{parseInt(h.slice(5,7),16)}}`;
}}

function draw(){{
  const p=PAGES[cur],sc=img.clientWidth/p.w;
  cv.width=img.clientWidth;cv.height=img.clientHeight||(p.h*sc);
  ctx.clearRect(0,0,cv.width,cv.height);
  p.elements.forEach(el=>{{
    if(hidden.has(el.label))return;
    const r=rgb(el.color),x=el.x*sc,y=el.y*sc,w=el.w*sc,h=el.h*sc,ih=(hov===el);
    ctx.fillStyle=ih?`rgba(${{r}},.35)`:`rgba(${{r}},.12)`;
    ctx.fillRect(x,y,w,h);
    ctx.strokeStyle=ih?`rgba(${{r}},1)`:`rgba(${{r}},.75)`;
    ctx.lineWidth=ih?2:1.2;ctx.strokeRect(x,y,w,h);
    ctx.font="bold 9px 'IBM Plex Mono',monospace";
    const tw=ctx.measureText(el.label).width;
    ctx.fillStyle=`rgba(${{r}},.9)`;ctx.fillRect(x,y,tw+8,14);
    ctx.fillStyle="#fff";ctx.fillText(el.label,x+4,y+10);
  }});
}}

function buildLeg(){{
  const p=PAGES[cur];leg.innerHTML="";
  const cnt={{}};p.elements.forEach(e=>{{cnt[e.label]=(cnt[e.label]||0)+1;}});
  p.legend.forEach(it=>{{
    const d=document.createElement("div");
    d.className="li"+(hidden.has(it.label)?" muted":"");
    d.innerHTML=`<div class="ld" style="background:${{it.color}}"></div>
      <span class="ll">${{it.label}}</span><span class="lc">${{cnt[it.label]||0}}</span>`;
    d.addEventListener("click",()=>{{
      hidden.has(it.label)?hidden.delete(it.label):hidden.add(it.label);
      d.classList.toggle("muted");draw();
    }});
    leg.appendChild(d);
  }});
  const total=p.elements.length,
        wt=p.elements.filter(e=>e.text.trim()).length,
        wc=p.elements.filter(e=>e.conf!=="N/A").length;
  stb.innerHTML=`Elementy: <b>${{total}}</b><br>S textom: <b>${{wt}}</b><br>S confidence: <b>${{wc}}</b>`;
}}

function load(i){{
  cur=i;const p=PAGES[i];hidden.clear();hov=null;
  img.src=p.b64;img.onload=()=>draw();
  pi2.textContent=p.name;pc.textContent=`${{i+1}} / ${{PAGES.length}}`;
  bP.disabled=(i===0);bN.disabled=(i===PAGES.length-1);ps.value=i;
  buildLeg();if(img.complete)draw();
}}

const cw=document.getElementById("cw");
cw.addEventListener("mousemove",e=>{{
  const p=PAGES[cur],rect=img.getBoundingClientRect(),sc=img.clientWidth/p.w;
  const mx=e.clientX-rect.left,my=e.clientY-rect.top;
  let found=null;
  for(let i=p.elements.length-1;i>=0;i--){{
    const el=p.elements[i];if(hidden.has(el.label))continue;
    if(mx>=el.x*sc&&mx<=(el.x+el.w)*sc&&my>=el.y*sc&&my<=(el.y+el.h)*sc){{found=el;break;}}
  }}
  if(found!==hov){{hov=found;draw();}}
  if(found){{
    ttl.textContent=found.label;ttl.style.color=found.color;
    ttt.textContent=found.text||"(žiadny text)";
    ttc.textContent=`confidence: ${{found.conf}}`;
    tt.style.display="block";
    let tx=e.clientX+16,ty=e.clientY+10;
    if(tx+420>window.innerWidth)tx=e.clientX-430;
    if(ty+180>window.innerHeight)ty=e.clientY-150;
    tt.style.left=tx+"px";tt.style.top=ty+"px";
  }}else{{tt.style.display="none";}}
}});
cw.addEventListener("mouseleave",()=>{{tt.style.display="none";hov=null;draw();}});
window.addEventListener("resize",draw);
bP.addEventListener("click",()=>{{if(cur>0)load(cur-1);}});
bN.addEventListener("click",()=>{{if(cur<PAGES.length-1)load(cur+1);}});
ps.addEventListener("change",e=>load(parseInt(e.target.value)));
document.addEventListener("keydown",e=>{{
  if(e.key==="ArrowLeft"&&cur>0)load(cur-1);
  if(e.key==="ArrowRight"&&cur<PAGES.length-1)load(cur+1);
}});
load(0);
</script>
</body>
</html>"""

    out = out_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


# ════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════

def main():
    png_folder = Path(PNG_FOLDER)
    out_json   = Path(OUT_JSON)
    out_vis    = Path(OUT_VIS)
    out_json.mkdir(parents=True, exist_ok=True)
    out_vis.mkdir(parents=True, exist_ok=True)

    png_files = sorted(png_folder.glob("*.png"))
    if MAX_IMAGES:
        png_files = png_files[:MAX_IMAGES]
    if not png_files:
        print(f"❌  Žiadne PNG v {png_folder}"); sys.exit(1)

    print(f"📂  {len(png_files)} obrázkov")

    all_results = []
    page_data   = []

    for idx, img_path in enumerate(png_files):
        print(f"\n[{idx+1}/{len(png_files)}] ⏳  {img_path.name}")
        try:
            result = run_dla_on_image(img_path)
        except Exception as e:
            print(f"   ❌  {e}"); continue

        nodes, pw, ph = extract_nodes(result)
        print(f"   ✅  {len(nodes)} nodov  (stránka {pw:.0f}×{ph:.0f})")

        save_json(nodes, img_path, pw, ph, out_json)

        all_results.append({"img_name": img_path.name, "nodes": nodes})

        with PILImage.open(img_path) as im:
            img_w, img_h = im.size

        page_data.append({
            "img_name": img_path.name,
            "b64":      img_to_b64(img_path),
            "img_w":    img_w, "img_h": img_h,
            "pw": pw,   "ph": ph,
            "nodes":    nodes,
        })

    if not all_results:
        print("❌  Nič spracované."); sys.exit(1)

    print("\n🌐  HTML viewer …")
    html_path = build_html(page_data, out_vis)
    print(f"   {html_path}")

    print("📊  Diagnostics …")
    diag_path = build_diagnostics(all_results, out_vis)
    print(f"   {diag_path}")

    print(f"\n🎉  Hotovo!  JSON: {out_json}/  |  HTML: {html_path}")


if __name__ == "__main__":
    main()