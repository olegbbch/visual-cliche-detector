
from __future__ import annotations
import io, os, math
import requests
from urllib.parse import quote_plus
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, ImageFilter

try:
    import cairosvg  # type: ignore
    _HAS_CAIRO = True
except Exception:
    _HAS_CAIRO = False

CATEGORIES = ["fintech","saas","wellness","kids","culture","fashion","ngo"]

@dataclass
class Features:
    dhash: int
    sym: float
    circularity: float
    stroke_like: float
    holes: int
    edge_density: float

def load_image(file_bytes: bytes, filename: str) -> Image.Image:
    name = (filename or "").lower()
    if name.endswith(".svg"):
        if not _HAS_CAIRO:
            raise RuntimeError("SVG upload requires cairosvg. Install it or upload PNG/JPG.")
        png_bytes = cairosvg.svg2png(bytestring=file_bytes, output_width=512, output_height=512)
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    return Image.open(io.BytesIO(file_bytes)).convert("RGBA")

def preprocess(img: Image.Image, size: int = 256) -> Image.Image:
    img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, (255,255,255,255))
    flat = Image.alpha_composite(bg, img).convert("RGB")

    flat = ImageOps.contain(flat, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255,255,255))
    x = (size - flat.size[0]) // 2
    y = (size - flat.size[1]) // 2
    canvas.paste(flat, (x,y))
    return canvas

def _dhash(img: Image.Image, hash_size: int = 16) -> int:
    gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(bool(b))
    return int(h)

def _hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())

def _binarize(img: Image.Image) -> np.ndarray:
    g = img.convert("L")
    arr = np.asarray(g, dtype=np.float32) / 255.0
    t = float(np.median(arr))
    bw = (arr < t).astype(np.uint8)
    return bw

def _count_holes(bw: np.ndarray) -> int:
    h, w = bw.shape
    visited = np.zeros_like(bw, dtype=np.uint8)

    def neighbors(y,x):
        for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
            ny,nx=y+dy,x+dx
            if 0<=ny<h and 0<=nx<w:
                yield ny,nx

    stack=[]
    for x in range(w):
        stack.append((0,x)); stack.append((h-1,x))
    for y in range(h):
        stack.append((y,0)); stack.append((y,w-1))
    while stack:
        y,x=stack.pop()
        if visited[y,x]:
            continue
        if bw[y,x]==0:
            visited[y,x]=1
            for ny,nx in neighbors(y,x):
                if not visited[ny,nx] and bw[ny,nx]==0:
                    stack.append((ny,nx))

    holes=0
    for y in range(h):
        for x in range(w):
            if bw[y,x]==0 and visited[y,x]==0:
                holes += 1
                stack=[(y,x)]
                while stack:
                    cy,cx=stack.pop()
                    if visited[cy,cx]==2:
                        continue
                    if bw[cy,cx]==0 and visited[cy,cx]==0:
                        visited[cy,cx]=2
                        for ny,nx in neighbors(cy,cx):
                            if bw[ny,nx]==0 and visited[ny,nx]==0:
                                stack.append((ny,nx))
    return int(holes)

def extract_features(img: Image.Image) -> Features:
    p = preprocess(img, 256)
    dh = _dhash(p, 16)

    g = p.convert("L")
    arr = np.asarray(g, dtype=np.float32) / 255.0
    flip = np.asarray(ImageOps.mirror(g), dtype=np.float32) / 255.0
    sym = 1.0 - float(np.mean(np.abs(arr - flip)))

    edges = g.filter(ImageFilter.FIND_EDGES)
    e = np.asarray(edges, dtype=np.float32) / 255.0
    edge_density = float(np.mean(e > 0.2))

    bw = _binarize(p)
    bw_img = Image.fromarray((bw*255).astype(np.uint8), mode="L")
    bw_edges = bw_img.filter(ImageFilter.FIND_EDGES)
    per = float(np.sum(np.asarray(bw_edges) > 10)) + 1e-6
    area = float(np.sum(bw)) + 1e-6

    circularity = float(4.0 * math.pi * area / (per * per))
    stroke_like = float(min(1.0, (per / (area + 1e-6)) / 0.20))
    holes = _count_holes(bw)

    return Features(
        dhash=dh,
        sym=max(0.0, min(1.0, sym)),
        circularity=max(0.0, min(1.0, circularity)),
        stroke_like=max(0.0, min(1.0, stroke_like)),
        holes=max(0, holes),
        edge_density=max(0.0, min(1.0, edge_density)),
    )

def similarity_to_set(f: Features, refs: List[Features]) -> Tuple[float, Optional[int]]:
    if not refs:
        return 0.0, None
    best_i = None
    best_sim = -1.0
    MAXBITS = 16*16
    for i, r in enumerate(refs):
        d = _hamming(f.dhash, r.dhash)
        sim = 1.0 - (d / MAXBITS)
        if sim > best_sim:
            best_sim = sim
            best_i = i
    return float(best_sim * 100.0), best_i

def detect_cliches(f: Features) -> List[Dict[str, str]]:
    signals = []
    if f.sym > 0.86:
        signals.append({"title":"High symmetry","desc":"Strong symmetry; many categories overuse symmetric tech marks.","common_in":"SaaS / Fintech / AI tools"})
    if f.circularity > 0.65 and f.holes >= 1:
        signals.append({"title":"Circle/loop archetype","desc":"High circularity + enclosed space suggests a loop/ring family mark.","common_in":"Wellness / Tech / Community brands"})
    if f.stroke_like > 0.75:
        signals.append({"title":"Monoline / stroke-like mark","desc":"High perimeter-to-area indicates a line-based mark (often trendy, often saturated).","common_in":"Wellness / DTC / Apps"})
    if f.holes >= 2:
        signals.append({"title":"Multi-hole geometry","desc":"Multiple enclosed regions often resemble 'infinity', 'link', or 'interlock' families.","common_in":"Fintech / Logistics / SaaS"})
    if f.edge_density < 0.04:
        signals.append({"title":"Ultra-minimal silhouette","desc":"Very low edge detail can drift into generic minimal icon territory.","common_in":"SaaS / Fashion / Lifestyle"})
    return signals[:6]

def trend_risk(f: Features) -> Dict[str, str]:
    if f.stroke_like > 0.78 and f.sym > 0.80:
        return {"status":"Saturated","note":"Monoline + symmetric geometry is heavily used in recent years."}
    if f.circularity > 0.70 and f.holes >= 1:
        return {"status":"Peak","note":"Loop/ring marks had a strong 2021–2024 run; now depend on category."}
    if f.edge_density > 0.12 and f.sym < 0.65:
        return {"status":"Rising","note":"More expressive, less symmetric forms are trending up in several categories."}
    return {"status":"Neutral","note":"No strong trend signal detected from shape stats."}

_EXPECTED = {
    "warm":{"sym_max":0.90},
    "human":{"sym_max":0.85},
    "playful":{"sym_max":0.92},
    "radical":{"sym_max":0.78},
    "bold":{"sym_max":0.82},
    "premium":{"stroke_max":0.75},
    "trust":{"sym_min":0.70},
    "tech":{"sym_min":0.75},
}

def semantic_mismatch(f: Features, keywords: List[str]) -> Optional[Dict[str, str]]:
    if not keywords:
        return None
    kws=[k.strip().lower() for k in keywords if k.strip()]
    if not kws:
        return None
    notes=[]
    for k in kws:
        rule=_EXPECTED.get(k)
        if not rule:
            continue
        if "sym_max" in rule and f.sym > rule["sym_max"]:
            notes.append(f"'{k}': often benefits from less rigid symmetry; current symmetry={f.sym:.2f}.")
        if "sym_min" in rule and f.sym < rule["sym_min"]:
            notes.append(f"'{k}': often correlates with higher symmetry; current symmetry={f.sym:.2f}.")
        if "stroke_max" in rule and f.stroke_like > rule["stroke_max"]:
            notes.append(f"'{k}': may suffer from overly thin/monoline marks; stroke_like={f.stroke_like:.2f}.")
    if not notes:
        return {"status":"No obvious mismatch","note":"Heuristic check found no strong tension with provided keywords."}
    return {"status":"Possible semantic tension","note":" ".join(notes)[:450]}

def load_reference_features(data_dir: str, category: str):
    cat_dir = os.path.join(data_dir, category)
    if not os.path.isdir(cat_dir):
        return [], []
    feats=[]
    names=[]
    for fn in sorted(os.listdir(cat_dir)):
        if not fn.lower().endswith((".png",".jpg",".jpeg",".webp",".svg")):
            continue
        path=os.path.join(cat_dir, fn)
        try:
            with open(path,"rb") as f:
                img=load_image(f.read(), fn)
            feats.append(extract_features(img))
            names.append(fn)
        except Exception:
            continue
    return feats, names
import os
import requests

SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def serpapi_reverse_image_search(image_url: str, num: int = 20):
    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": SERPAPI_KEY,
        "hl": "en",
        "num": num,
    }

    r = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    results = []

    for item in data.get("visual_matches", [])[:num]:
        results.append({
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "thumbnail": item.get("thumbnail", ""),
            "source": item.get("source", ""),
        })

    return results
def world_scan(img, max_results: int = 8) -> list[dict]:
    """
    World scan via SerpAPI (Google Lens).
    Returns list of {title, link, score}.
    Needs SERPAPI_KEY in env (Streamlit Secrets).
    """
    api_key = os.getenv("SERPAPI_KEY", "").strip()
    if not api_key:
        return []

    # Convert PIL image to bytes (PNG)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    # Upload to Cloudinary (unsigned)
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", "").strip()
    upload_preset = os.getenv("CLOUDINARY_UPLOAD_PRESET", "").strip()
    if not cloud_name or not upload_preset:
        return []

    upload_url = f"https://api.cloudinary.com/v1_1/{cloud_name}/image/upload"
    files = {"file": ("image.png", image_bytes, "image/png")}
    data = {"upload_preset": upload_preset}

    r = requests.post(upload_url, files=files, data=data, timeout=30)
    r.raise_for_status()
    image_url = r.json().get("secure_url", "")
    if not image_url:
        return []

    # SerpAPI Google Lens
    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": api_key,
    }
    rr = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
    rr.raise_for_status()
    j = rr.json()

    out = []

    # Try common result buckets
    items = []
    if isinstance(j.get("visual_matches"), list):
        items = j["visual_matches"]
    elif isinstance(j.get("inline_images"), list):
        items = j["inline_images"]

    for it in items[:max_results]:
        title = it.get("title", "") or it.get("source", "") or "Result"
        link = it.get("link", "") or it.get("source", "")

        thumb = (
            it.get("thumbnail")
            or it.get("thumbnail_url")
            or it.get("image")
            or it.get("image_url")
        )

        score = it.get("score", 0.0)

        out.append({
            "title": title,
            "link": link,
            "score": score,
            "thumbnail": thumb
        })

    return out
