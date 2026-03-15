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
            font-size: 2.15rem !important;
            margin-top: 0 !important;
            margin-bottom: 0.35rem !important;
            line-height: 1.05 !important;
        }

        h2 {
            font-weight: 700 !important;
            font-size: 1.42rem !important;
            margin-top: 0.2rem !important;
            margin-bottom: 0.7rem !important;
        }

        h3 {
            font-weight: 700 !important;
            font-size: 1.08rem !important;
        }

        p, li, div[data-testid="stMarkdownContainer"] {
            color: var(--text);
            font-size: 1rem;
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
            min-height: 420px !important;
            height: 420px !important;
            padding: 1.8rem 1.4rem !important;
            box-shadow: var(--shadow);
            transition: all 0.18s ease;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        div[data-testid="stFileUploaderDropzone"]:hover {
            border-color: #97a9bf !important;
            background: #fcfdff;
        }

        div[data-testid="stFileUploaderDropzone"] > div {
            width: 100% !important;
            height: 100% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        div[data-testid="stFileUploaderDropzoneInstructions"] {
            width: 100% !important;
            height: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            gap: 0.35rem !important;
        }

        div[data-testid="stFileUploaderDropzone"] * {
            font-size: 1.04rem !important;
        }

        div[data-testid="stFileUploaderDropzone"] small {
            color: var(--muted) !important;
            font-size: 0.98rem !important;
        }

        div[data-testid="stFileUploaderDropzone"] button {
            border-radius: 999px !important;
            min-height: 52px !important;
            padding: 0.9rem 1.28rem !important;
            border: 1px solid #d3d9e2 !important;
            background: white !important;
            color: var(--text) !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
        }

        div[data-testid="stButton"] > button {
            width: 100%;
            min-height: 60px;
            border-radius: 16px;
            border: 1px solid #111111 !important;
            background: #111111 !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-size: 1.02rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em;
            box-shadow: none;
            transition: transform 0.12s ease, background 0.12s ease;
        }

        div[data-testid="stButton"] > button:hover {
            transform: translateY(-1px);
            background: #222222 !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        div[data-testid="stButton"] > button:disabled {
            opacity: 0.42;
            background: #111111 !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
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
        <div style="border:1px solid #e5e7eb;border-radius:20px;padding:16px">

        <img src="{thumb}" style="width:100%;height:180px;object-fit:contain">

        <div style="font-weight:700;margin-top:8px">{title}</div>

        <div style="color:#6b7280;font-size:.85rem">{domain}</div>

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
