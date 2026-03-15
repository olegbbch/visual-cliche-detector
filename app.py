import base64
import hashlib
import io
import os
from urllib.parse import urlparse

import requests
import streamlit as st
from PIL import Image

from vcd_utils import (
    detect_cliches,
    extract_features,
    load_image,
    similarity_to_set,
    trend_risk,
    world_scan,
)

SCAN_POOL = 12
VISIBLE_MATCHES = 3
WARNING_POOL = 5

APP_TITLE = "LogoRadar"


def inject_ui_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #f5f7fb;
            --surface: #ffffff;
            --line: #dbe5f0;
            --text: #111111;
            --muted: #5f6673;
            --green: #15803d;
            --green-soft: #eefbf3;
            --red: #dc2626;
            --red-soft: #fff2f2;
            --amber: #b45309;
            --amber-soft: #fff8eb;
            --shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: linear-gradient(180deg, #f8f9fc 0%, #f3f5f9 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1580px;
            padding-top: 3rem;
            padding-bottom: 2.2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        h1 {
            font-weight: 800 !important;
            font-size: 2.3rem !important;
            margin-top: 0 !important;
            margin-bottom: 0.35rem !important;
            line-height: 1.05 !important;
        }

        h2 {
            font-weight: 500 !important;
            font-size: 1.56rem !important;
            margin-top: 0.2rem !important;
            margin-bottom: 0.7rem !important;
        }

        h3 {
            font-weight: 500 !important;
            font-size: 1.72rem !important;
        }

        p, li, div[data-testid="stMarkdownContainer"] {
            color: var(--text);
            font-size: 1.08rem;
        }

        div[data-testid="stHorizontalBlock"] {
            gap: 1.6rem;
        }

        div[data-testid="stFileUploader"] > label,
        div[data-testid="stButton"] > label {
            display: none !important;
        }

        div[data-testid="stFileUploader"] {
            width: 100%;
        }

        div[data-testid="stFileUploader"] section {
            min-height: unset !important;
            height: auto !important;
        }

        div[data-testid="stFileUploaderDropzone"] {
            background: rgba(255,255,255,0.95);
            border: 1.5px dashed #c6d2e1 !important;
            border-radius: 24px !important;
            min-height: 320px !important;
            height: 320px !important;
            padding: 1.8rem 1.4rem !important;
            box-shadow: var(--shadow);
            transition: all 0.18s ease;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        div[data-testid="stFileUploaderDropzone"] * {
            font-size: 1.14rem !important;
        }

        div[data-testid="stButton"] > button {
            width: 100%;
            min-height: 60px;
            border-radius: 16px;
            border: 1px solid #111111 !important;
            background: #111111 !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-size: 1.9rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em;
        }

        div[data-testid="stAlert"] {
            border-radius: 18px !important;
            border: 1px solid #d7e2ef !important;
            box-shadow: none !important;
        }

        .lr-brand-wrap {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 0.2rem;
            margin-top: 0.2rem;
        }

        .lr-brand-fallback {
            font-size: 2.15rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #111111;
        }

        .lr-subtitle {
            color: var(--muted);
            font-size: 1.12rem;
            margin-bottom: 1.2rem;
        }

        .lr-match-title {
            font-size: 1.18rem;
            font-weight: 700;
        }

        .lr-match-domain {
            font-size: 1rem;
        }

        .lr-source-link {
            font-size: 1rem;
        }

        .lr-disclaimer {
            color: #6b7280;
            font-size: 0.96rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def proximity_label(sim_pct):
    try:
        s = float(sim_pct)
    except Exception:
        return "Unknown"
    if s >= 60:
        return "High"
    if s >= 40:
        return "Medium"
    return "Low"


def proximity_badge_class(label: str) -> str:
    if label == "High":
        return "lr-badge-red"
    if label == "Medium":
        return "lr-badge-amber"
    if label == "Low":
        return "lr-badge-green"
    return "lr-badge-neutral"


def domain_of(url: str):
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def thumb_features(url):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGBA")
        return extract_features(im)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def cached_world_scan(file_bytes: bytes, file_name: str):
    img = load_image(file_bytes, file_name)
    return world_scan(img, max_results=SCAN_POOL)


st.set_page_config(page_title=APP_TITLE, layout="wide")
inject_ui_css()

st.markdown('<div class="lr-brand-fallback">LogoRadar</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="lr-subtitle">Early warning system for logo similarity, cliché signals and trend risk.</div>',
    unsafe_allow_html=True,
)

colL, colR = st.columns([1, 2], gap="large")

with colL:

    st.subheader("Input")

    up = st.file_uploader(
        "Upload logo mark (SVG/PNG/JPG)",
        type=["svg", "png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )

    run = st.button("Analyze", disabled=(up is None), use_container_width=True)

with colR:

    st.subheader("Report")

    if not up:
        st.info("Upload a mark to start.")

    elif run:

        file_bytes = up.getvalue()
        file_name = up.name

        img = load_image(file_bytes, file_name)
        f = extract_features(img)

        st.image(img, caption="Uploaded mark", width=300)

        with st.spinner("Searching the web for similar marks..."):
            world_results = cached_world_scan(file_bytes, file_name)

        match_data = []

        for r in world_results:
            thumb = r.get("thumbnail", "")
            title = r.get("title", "")
            link = r.get("link", "")

            sim_pct = None

            if thumb:
                tf = thumb_features(thumb)
                if tf is not None:
                    sim_pct, _ = similarity_to_set(f, [tf])

            label = proximity_label(sim_pct if sim_pct else 0)

            match_data.append(
                {
                    "thumb": thumb,
                    "title": title,
                    "link": link,
                    "label": label,
                }
            )

        st.subheader("World scan")

        cols = st.columns(3)

        for i, m in enumerate(match_data[:VISIBLE_MATCHES]):
            with cols[i % 3]:

                if m["thumb"]:
                    st.image(m["thumb"])

                st.write(m["title"])
                st.write(f"Proximity: {m['label']}")

                if m["link"]:
                    st.markdown(f"[Source]({m['link']})")

st.markdown(
    """
<div class="lr-disclaimer">
Disclaimer: This tool supports creative exploration and early visual checks.
It is not a legal trademark search and does not replace professional trademark clearance.
</div>
""",
    unsafe_allow_html=True,
)
