import io
import os

import numpy as np
import requests
import streamlit as st
from PIL import Image

from vcd_utils import (
    CATEGORIES,
    detect_cliches,
    extract_features,
    load_image,
    semantic_mismatch,
    trend_risk,
    world_scan,
)
from pdf_utils import make_risk_sheet_pdf


def proximity_label(sim_pct):
    try:
        s = float(sim_pct)
    except Exception:
        return "Unknown"

    if s < 0:
        return "Unknown"
    if s >= 75:
        return "High"
    if s >= 55:
        return "Medium"
    return "Low"


@st.cache_data(show_spinner=False)
def thumb_features(url: str):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGBA")
        return extract_features(im)
    except Exception:
        return None


APP_TITLE = "Visual Cliché Detector (MVP)"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(
    "Designer early warning system: web similarity + cliché signals + trend risk + optional semantic tension. "
    "(Heuristic; not legal/trademark search.)"
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
        st.info("Upload a mark to start. If SVG fails on cloud, upload PNG/JPG instead.")
    elif run:
        try:
            img = load_image(up.getvalue(), up.name)
            st.image(img, caption="Uploaded mark", width=260)

            f = extract_features(img)

            world_results = []
            if world_on:
                with st.spinner("World scan: searching the web…"):
                    world_results = world_scan(img, max_results=world_k)

            st.markdown("## Early warning")
            if world_results:
                st.warning("⚠️ Designer, be careful\n\nSimilar visual marks were found online.")
            else:
                st.success("✅ Looks safe\n\nNo strong visual similarities detected.")

            if world_on:
                st.markdown("## 🌐 World scan (web)")

                if not world_results:
                    st.info("No web matches found.")
                else:
                    cols = st.columns(3, gap="medium")

                    for i, r in enumerate(world_results[:3]):
                        with cols[i % 3]:
                            thumb = r.get("thumbnail", "")
                            title = r.get("title", "") or "Result"
                            link = r.get("link", "")

                            sim_pct = None
                            if thumb:
                                tf = thumb_features(thumb)
                                if tf is not None:
                                    a = np.array(f).reshape(-1)
                                    b = np.array(tf).reshape(-1)
                                    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
                                    sim_pct = float(np.dot(a, b) / denom) * 100.0

                            if thumb:
                                st.image(thumb, use_container_width=True)

                            label = proximity_label(sim_pct if sim_pct is not None else -1)

                            st.caption(title)
                            if sim_pct is None:
                                st.caption(f"Proximity: **{label}**")
                            else:
                                st.caption(f"Proximity: **{label}** ({sim_pct:.0f}%)")

                            if link:
                                st.markdown(f"[Source]({link})")

            st.markdown("## Cliché signals")
            cliches = detect_cliches(f)
            if not cliches:
                st.success("No strong cliché signals detected (heuristic).")
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
