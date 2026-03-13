import io
import os
from urllib.parse import urlparse

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageFilter, ImageOps

from vcd_utils import (
    CATEGORIES,
    detect_cliches,
    extract_features,
    load_image,
    semantic_mismatch,
    similarity_to_set,
    trend_risk,
    world_scan,
)
from pdf_utils import make_risk_sheet_pdf


def proximity_label(sim_pct):
    try:
        s = float(sim_pct)
    except Exception:
        return "Unknown"

    if s >= 70:
        return "High"
    if s >= 45:
        return "Medium"
    return "Low"


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def title_is_noise(title: str, link: str) -> bool:
    text = f"{title} {link}".lower()

    noisy_terms = [
        "youtube", "youtu.be", "reddit", "pinterest", "tiktok", "facebook",
        "instagram", "shutterstock", "freepik", "clipart", "meme",
        "tutorial", "how to", "exercise", "lyrics", "song", "video",
        "drawing", "sketch", "doodle", "cartoon", "illustration", "wallpaper",
        "squat", "flexibility", "coloring", "printable", "svg download",
        "stitch guide", "diagram", "schematic", "process", "step by step",
    ]

    noisy_domains = [
        "youtube.com", "youtu.be", "reddit.com", "pinterest.com", "tiktok.com",
        "facebook.com", "instagram.com", "shutterstock.com", "freepik.com",
    ]

    if any(term in text for term in noisy_terms):
        return True

    dom = domain_of(link)
    if any(nd in dom for nd in noisy_domains):
        return True

    return False


def title_is_logo_like(title: str, link: str) -> bool:
    text = f"{title} {link}".lower()

    logo_terms = [
        "logo", "brand", "branding", "identity", "company", "official",
        "inc", "ltd", "llc", "group", "studio", "agency", "services",
        "hospitality", "design", "projects", "portfolio", "case study",
        "help center", "about us", "our work",
    ]

    brandy_domains = [
        "1000logos", "behance", "dribbble", "brandsoftheworld",
        "logowik", "logos-world", "crunchbase", "linkedin",
    ]

    if any(term in text for term in logo_terms):
        return True

    dom = domain_of(link)
    if any(bd in dom for bd in brandy_domains):
        return True

    return False


@st.cache_data(show_spinner=False)
def download_thumb_bytes(url: str):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def image_to_png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def open_rgba_from_bytes(data: bytes):
    try:
        return Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception:
        return None


def normalize_logo_image(im: Image.Image, size: int = 256) -> Image.Image:
    # Put onto white background to stabilize transparency
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    merged = Image.alpha_composite(bg, im.convert("RGBA")).convert("L")
    merged = ImageOps.autocontrast(merged)
    merged = ImageOps.contain(merged, (size, size))

    canvas = Image.new("L", (size, size), 255)
    x = (size - merged.width) // 2
    y = (size - merged.height) // 2
    canvas.paste(merged, (x, y))
    return canvas


def foreground_mask(im: Image.Image, size: int = 256) -> np.ndarray:
    gray = normalize_logo_image(im, size=size)
    arr = np.asarray(gray).astype(np.float32)

    # Assume logo strokes are darker than background
    threshold = np.percentile(arr, 75)
    mask = arr < threshold

    # If almost everything is foreground/background, fallback to mean threshold
    ratio = mask.mean()
    if ratio < 0.03 or ratio > 0.80:
        threshold = arr.mean()
        mask = arr < threshold

    return mask.astype(np.float32)


def silhouette_similarity(im1: Image.Image, im2: Image.Image) -> float:
    m1 = foreground_mask(im1)
    m2 = foreground_mask(im2)

    intersection = np.logical_and(m1 > 0.5, m2 > 0.5).sum()
    union = np.logical_or(m1 > 0.5, m2 > 0.5).sum()

    if union == 0:
        return 0.0

    iou = intersection / union
    return float(iou * 100.0)


def edge_similarity(im1: Image.Image, im2: Image.Image, size: int = 256) -> float:
    g1 = normalize_logo_image(im1, size=size).filter(ImageFilter.FIND_EDGES)
    g2 = normalize_logo_image(im2, size=size).filter(ImageFilter.FIND_EDGES)

    a = np.asarray(g1).astype(np.float32).reshape(-1)
    b = np.asarray(g2).astype(np.float32).reshape(-1)

    a = a - a.mean()
    b = b - b.mean()

    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0

    cos = float(np.dot(a, b) / denom)
    cos = max(-1.0, min(1.0, cos))
    return max(0.0, cos) * 100.0


def combined_proximity(uploaded_im: Image.Image, uploaded_features, thumb_im: Image.Image, thumb_features) -> float:
    feature_sim, _ = similarity_to_set(uploaded_features, [thumb_features])
    sil_sim = silhouette_similarity(uploaded_im, thumb_im)
    edg_sim = edge_similarity(uploaded_im, thumb_im)

    # Weighted blend: structure matters more than raw thumbnail appearance
    score = (0.45 * feature_sim) + (0.35 * sil_sim) + (0.20 * edg_sim)
    return float(max(0.0, min(100.0, score)))


def warning_state(matches):
    relevant = [m for m in matches if m.get("is_relevant", False)]

    high_count = sum(1 for m in relevant if m.get("label") == "High")
    medium_count = sum(1 for m in relevant if m.get("label") == "Medium")

    if high_count >= 1:
        return "high"
    if medium_count >= 2:
        return "mixed"
    return "safe"


def build_match_data(result, uploaded_im, uploaded_features):
    thumb = result.get("thumbnail", "")
    title = result.get("title", "") or "Result"
    link = result.get("link", "")

    sim_pct = None
    if thumb:
        thumb_bytes = download_thumb_bytes(thumb)
        if thumb_bytes:
            thumb_im = open_rgba_from_bytes(thumb_bytes)
            if thumb_im is not None:
                try:
                    tf = extract_features(thumb_im)
                    sim_pct = combined_proximity(uploaded_im, uploaded_features, thumb_im, tf)
                except Exception:
                    sim_pct = None

    label = proximity_label(sim_pct if sim_pct is not None else 0)
    is_noise = title_is_noise(title, link)
    is_logo_like = title_is_logo_like(title, link)
    is_relevant = (not is_noise) and is_logo_like

    return {
        "thumb": thumb,
        "title": title,
        "link": link,
        "sim_pct": sim_pct,
        "label": label,
        "is_noise": is_noise,
        "is_logo_like": is_logo_like,
        "is_relevant": is_relevant,
    }


def render_match_card(match):
    if match["thumb"]:
        st.image(match["thumb"], use_container_width=True)

    st.caption(match["title"])

    if match["sim_pct"] is None:
        st.caption(f"Proximity: **{match['label']}**")
    else:
        st.caption(f"Proximity: **{match['label']}** ({match['sim_pct']:.0f}%)")

    if match["link"]:
        st.markdown(f"[Source]({match['link']})")


APP_TITLE = "Visual Cliché Detector (MVP)"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(
    "Designer early warning system: web similarity + cliché signals + trend risk + optional semantic tension."
)

data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)

colL, colR = st.columns([1, 2], gap="large")

with colL:
    st.subheader("Input")

    up = st.file_uploader(
        "Upload logo mark (SVG/PNG/JPG)",
        type=["svg", "png", "jpg", "jpeg", "webp"],
    )

    category = st.selectbox("Category context", options=CATEGORIES, index=0)
    extra_cat = st.selectbox("Optional second context", options=["-"] + CATEGORIES, index=0)
    keywords = st.text_input(
        "Positioning keywords (comma-separated, optional)",
        placeholder="human, warm, bold",
    )

    world_on = True
    world_k = st.slider("How many web matches?", 3, 15, 5)

    st.markdown("---")
    run = st.button("Analyze", type="primary", disabled=(up is None))

with colR:
    st.subheader("Report")

    if not up:
        st.info("Upload a mark to start.")
    elif run:
        try:
            img = load_image(up.getvalue(), up.name)
            st.image(img, caption="Uploaded mark", width=260)

            f = extract_features(img)

            world_results = []
            if world_on:
                with st.spinner("World scan: searching the web…"):
                    world_results = world_scan(img, max_results=world_k)

            match_data = [build_match_data(r, img, f) for r in world_results]
            relevant_matches = [m for m in match_data if m.get("is_relevant", False)]
            state = warning_state(match_data)

            st.markdown("## Early warning")
            if state == "high":
                st.warning(
                    "⚠️ Designer, be careful\n\nStrong logo-like visual overlap was found online."
                )
            elif state == "mixed":
                st.info(
                    "🟡 Mixed signal\n\nSome potentially relevant logo-like matches were found. Review manually."
                )
            else:
                st.success(
                    "✅ Looks safe\n\nNo meaningful logo-like matches found online."
                )

            st.markdown("## 🌐 World scan (web)")
            if not relevant_matches:
                st.info("No logo-like matches found.")
            else:
                visible_matches = relevant_matches[:3]
                extra_matches = relevant_matches[3:world_k]

                cols = st.columns(3, gap="medium")
                for i, match in enumerate(visible_matches):
                    with cols[i % 3]:
                        render_match_card(match)

                if extra_matches:
                    with st.expander(f"More logo-like matches ({len(extra_matches)})"):
                        more_cols = st.columns(3, gap="medium")
                        for i, match in enumerate(extra_matches):
                            with more_cols[i % 3]:
                                render_match_card(match)

            st.markdown("## Cliché signals")
            cliches = detect_cliches(f)
            if not cliches:
                st.success("No strong cliché signals detected.")
            else:
                for s in cliches:
                    with st.expander(s["title"], expanded=True):
                        st.write(s["desc"])
                        st.caption(f"Common in: {s['common_in']}")

            st.markdown("## Trend risk")
            tr = trend_risk(f)
            st.write(f"**Status:** {tr['status']}")
            st.write(tr["note"])

            st.markdown("## Semantic check")
            kw = [k.strip() for k in keywords.split(",")] if keywords else []
            sem = semantic_mismatch(f, kw)
            if sem:
                st.write(f"**{sem['status']}**")
                st.write(sem["note"])
            else:
                st.caption("No positioning keywords provided.")

            st.markdown("## Export")
            if st.button("Download PDF Risk Sheet"):
                pdf_bytes = make_risk_sheet_pdf(
                    title=APP_TITLE,
                    category=category,
                    extra_category=(extra_cat if extra_cat != "-" else None),
                    similarity_rows=[],
                    cliches=cliches,
                    trend=tr,
                    semantic=sem,
                )
                st.download_button(
                    "Click to download",
                    data=pdf_bytes,
                    file_name="vcd_risk_sheet.pdf",
                    mime="application/pdf",
                )

        except Exception as e:
            st.error(f"Error while analyzing: {e}")
            st.stop()
