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

APP_TITLE = "LogoRadar"

SCAN_POOL = 12
VISIBLE_MATCHES = 3
WARNING_POOL = 5


def inject_ui_css():
    st.markdown(
        """
        <style>

        html, body, [class*="css"] {
            font-family: Inter, system-ui, sans-serif;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 2.5rem;
            padding-bottom: 2rem;
        }

        h1 {
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        /* DROPZONE */

        div[data-testid="stFileUploaderDropzone"] {
            background: white;
            border: 1.5px dashed #cbd5e1 !important;
            border-radius: 24px !important;

            min-height: 420px !important;
            height: 420px !important;

            display: flex !important;
            align-items: center !important;
            justify-content: center !important;

            padding: 1.5rem !important;

            transition: all .2s ease;
        }

        div[data-testid="stFileUploaderDropzone"]:hover {
            border-color: #64748b !important;
            background: #fafafa;
        }

        /* UPLOAD BUTTON */

        div[data-testid="stFileUploaderDropzone"] button {
            border-radius: 999px !important;
            border: 1px solid #e2e8f0 !important;
            padding: .6rem 1.2rem !important;
        }

        /* ANALYZE BUTTON */

        div[data-testid="stButton"] > button {
            width: 100%;
            min-height: 60px;

            border-radius: 16px;

            background: #111 !important;
            border: 1px solid #111 !important;

            color: white !important;
            font-weight: 700;
            font-size: 1rem;

            transition: all .15s ease;
        }

        div[data-testid="stButton"] > button:hover {
            background: #222 !important;
        }

        div[data-testid="stButton"] > button:disabled {
            opacity: .4;
        }

        /* MATCH CARDS */

        .match-card {
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 14px;
            background: white;
        }

        .match-title {
            font-weight: 600;
            font-size: 0.95rem;
            margin-top: 8px;
        }

        .match-domain {
            color: #6b7280;
            font-size: 0.8rem;
        }

        /* DISCLAIMER */

        .disclaimer {
            font-size: .8rem;
            color: #6b7280;
            margin-top: 50px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def file_fingerprint(file_bytes: bytes, file_name: str) -> str:
    h = hashlib.sha256()
    h.update(file_name.encode("utf-8", errors="ignore"))
    h.update(file_bytes)
    return h.hexdigest()


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


def domain_of(url: str):
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def thumb_features(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGBA")
        return extract_features(im)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def cached_world_scan(file_bytes: bytes, file_name: str):
    img = load_image(file_bytes, file_name)
    return world_scan(img, max_results=SCAN_POOL)


def build_match_data(result, uploaded_features):
    thumb = result.get("thumbnail", "")
    title = result.get("title", "") or "Result"
    link = result.get("link", "")

    sim_pct = None
    if thumb:
        tf = thumb_features(thumb)
        if tf is not None:
            sim_pct, _ = similarity_to_set(uploaded_features, [tf])

    label = proximity_label(sim_pct if sim_pct is not None else 0)

    return {
        "thumb": thumb,
        "title": title,
        "link": link,
        "label": label,
        "sim_pct": sim_pct,
    }


def render_match_card(match):

    thumb = match["thumb"]
    title = match["title"]
    link = match["link"]
    label = match["label"]

    domain = domain_of(link)

    st.markdown(
        f"""
        <div class="match-card">

        <img src="{thumb}" style="width:100%;height:160px;object-fit:contain">

        <div class="match-title">{title}</div>

        <div class="match-domain">{domain}</div>

        <div style="margin-top:6px;font-weight:600">
        Proximity: {label}
        </div>

        <div style="margin-top:6px">
        <a href="{link}" target="_blank">Source</a>
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title=APP_TITLE, layout="wide")

inject_ui_css()

st.title("LogoRadar")

st.caption(
    "Early warning system for logo similarity, cliché signals and trend risk."
)

colL, colR = st.columns([1, 2], gap="large")

with colL:

    st.subheader("Input")

    up = st.file_uploader(
        "Upload logo mark",
        type=["svg", "png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )

    run = st.button(
        "Analyze",
        disabled=(up is None),
        use_container_width=True,
    )


with colR:

    st.subheader("Report")

    if not up:
        st.info("Upload a mark to start.")

    elif run:

        try:

            file_bytes = up.getvalue()
            file_name = up.name

            img = load_image(file_bytes, file_name)
            st.image(img, caption="Uploaded mark", width=260)

            f = extract_features(img)

            scan_placeholder = st.empty()

            with scan_placeholder.container():

                st.info(
                    "Web scan in progress…\n\nSearching the web for similar marks."
                )

            st.markdown("## Cliché signals")

            cliches = detect_cliches(f)

            if not cliches:
                st.success("No strong cliché signals detected.")
            else:
                for s in cliches:
                    st.write("**" + s["title"] + "**")
                    st.write(s["desc"])
                    st.caption("Common in: " + s["common_in"])

            st.markdown("## Trend risk")

            tr = trend_risk(f)

            st.write("**Status:**", tr["status"])
            st.write(tr["note"])

            world_results = cached_world_scan(file_bytes, file_name)

            match_data = [build_match_data(r, f) for r in world_results]

            with scan_placeholder.container():

                st.markdown("## World scan")

                if not match_data:

                    st.info("No logo-like matches found.")

                else:

                    cols = st.columns(3)

                    for i, m in enumerate(match_data[:VISIBLE_MATCHES]):

                        with cols[i % 3]:

                            render_match_card(m)

        except Exception as e:

            st.error(f"Error while analyzing: {e}")

st.markdown(
    """
    <div class="disclaimer">

    Disclaimer: This application is not a legal trademark search or patent check.
    It is a creative exploration tool designed to help designers detect visual similarity and cliché risks early in the design process.

    </div>
    """,
    unsafe_allow_html=True,
)
