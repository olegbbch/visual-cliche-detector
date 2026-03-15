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

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1580px;
}

h3 {
    font-weight: 500 !important;
    font-size: 1.6rem !important;
}

/* DROPZONE */

div[data-testid="stFileUploaderDropzone"] {
    min-height: 320px !important;
    height: 320px !important;
}

/* ANALYZE BUTTON */

div[data-testid="stButton"] > button {

    width: 100%;
    min-height: 60px;

    border-radius: 16px;

    border: 1px solid #111111 !important;
    background: #111111 !important;

    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;

    font-size: 1.32rem !important;
}

/* THIS IS THE IMPORTANT PART */

div[data-testid="stButton"] > button p {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
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

        r = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        r.raise_for_status()

        im = Image.open(io.BytesIO(r.content)).convert("RGBA")

        return extract_features(im)

    except Exception:

        return None


@st.cache_data(show_spinner=False)
def cached_world_scan(file_bytes: bytes, file_name: str):

    img = load_image(file_bytes, file_name)

    return world_scan(img, max_results=SCAN_POOL)


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
)

inject_ui_css()

st.title("LogoRadar")

st.markdown(
    "Early warning system for logo similarity, cliché signals and trend risk."
)

colL, colR = st.columns([1, 2])

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

        file_bytes = up.getvalue()
        file_name = up.name

        img = load_image(file_bytes, file_name)

        f = extract_features(img)

        st.image(
            img,
            caption="Uploaded mark",
            width=300,
        )

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
---

**Disclaimer**

This application is not a legal trademark search and does not replace professional trademark clearance.  
It is intended only as a **creative exploration tool** for designers.
"""
)
