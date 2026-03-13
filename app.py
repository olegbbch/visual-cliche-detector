import io
import os
from urllib.parse import urlparse

import requests
import streamlit as st
from PIL import Image

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


def title_is_noise(title: str, link: str):
    text = f"{title} {link}".lower()

    noisy_terms = [
        "youtube","youtu.be","reddit","pinterest","tiktok","facebook",
        "instagram","shutterstock","freepik","clipart","meme",
        "tutorial","how to","exercise","lyrics","song","video",
        "drawing","sketch","doodle","cartoon","illustration","wallpaper",
        "diagram","schematic","process","step by step"
    ]

    noisy_domains = [
        "youtube.com","youtu.be","reddit.com","pinterest.com",
        "tiktok.com","facebook.com","instagram.com"
    ]

    if any(t in text for t in noisy_terms):
        return True

    dom = domain_of(link)
    if any(d in dom for d in noisy_domains):
        return True

    return False


def title_is_logo_like(title: str, link: str):
    text = f"{title} {link}".lower()

    logo_terms = [
        "logo","brand","branding","identity","company","official",
        "inc","ltd","llc","group","studio","agency","services",
        "hospitality","design","projects","portfolio","case study",
        "help center","about us"
    ]

    brand_domains = [
        "1000logos","behance","dribbble","brandsoftheworld",
        "logowik","logos-world","crunchbase","linkedin"
    ]

    if any(t in text for t in logo_terms):
        return True

    dom = domain_of(link)
    if any(d in dom for d in brand_domains):
        return True

    return False


@st.cache_data(show_spinner=False)
def thumb_features(url):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGBA")
        return extract_features(im)
    except Exception:
        return None


def build_match_data(result, uploaded_features):

    thumb = result.get("thumbnail","")
    title = result.get("title","") or "Result"
    link = result.get("link","")

    sim_pct = None

    if thumb:
        tf = thumb_features(thumb)
        if tf is not None:
            sim_pct,_ = similarity_to_set(uploaded_features,[tf])

    label = proximity_label(sim_pct if sim_pct is not None else 0)

    is_noise = title_is_noise(title,link)
    is_logo_like = title_is_logo_like(title,link)

    is_relevant = (not is_noise) and is_logo_like

    return {
        "thumb":thumb,
        "title":title,
        "link":link,
        "label":label,
        "sim_pct":sim_pct,
        "is_relevant":is_relevant
    }


def warning_state(matches):

    relevant = [m for m in matches if m["is_relevant"]]

    high_count = sum(1 for m in relevant if m["label"]=="High")
    medium_count = sum(1 for m in relevant if m["label"]=="Medium")

    if high_count >= 1:
        return "high"

    if medium_count >= 2:
        return "mixed"

    return "safe"


def render_match_card(match):

    if match["thumb"]:
        st.image(match["thumb"], use_container_width=True)

    st.caption(match["title"])
    st.caption(f"Proximity: **{match['label']}**")

    if match["link"]:
        st.markdown(f"[Source]({match['link']})")


APP_TITLE = "Visual Cliché Detector (MVP)"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

st.caption(
"Designer early warning system: web similarity + cliché signals + trend risk + optional semantic tension."
)

data_dir = os.path.join(os.path.dirname(__file__),"data")
os.makedirs(data_dir,exist_ok=True)

colL,colR = st.columns([1,2],gap="large")


with colL:

    st.subheader("Input")

    up = st.file_uploader(
        "Upload logo mark (SVG/PNG/JPG)",
        type=["svg","png","jpg","jpeg","webp"]
    )

    category = st.selectbox("Category context", options=CATEGORIES, index=0)

    extra_cat = st.selectbox(
        "Optional second context",
        options=["-"] + CATEGORIES,
        index=0
    )

    keywords = st.text_input(
        "Positioning keywords (comma-separated, optional)",
        placeholder="human, warm, bold"
    )

    world_k = st.slider("How many web matches?",3,15,5)

    st.markdown("---")

    run = st.button("Analyze",type="primary",disabled=(up is None))


with colR:

    st.subheader("Report")

    if not up:
        st.info("Upload a mark to start.")

    elif run:

        try:

            img = load_image(up.getvalue(), up.name)
            st.image(img, caption="Uploaded mark", width=260)

            f = extract_features(img)

            with st.spinner("World scan: searching the web…"):
                world_results = world_scan(img, max_results=world_k)

            match_data = [build_match_data(r,f) for r in world_results]

            relevant_matches = [m for m in match_data if m["is_relevant"]]

            state = warning_state(match_data)

            st.markdown("## Early warning")

            if state=="high":
                st.warning(
                "⚠️ Designer, be careful\n\nStrong logo-like visual overlap was found online."
                )

            elif state=="mixed":
                st.info(
                "Some potentially relevant logo-like matches were found."
                )

            else:
                st.success(
                "Looks safe.\n\nNo meaningful logo-like matches found online."
                )

            st.markdown("## 🌐 World scan (web)")

            if not relevant_matches:

                st.info("No logo-like matches found.")

            else:

                visible = relevant_matches[:3]
                extra = relevant_matches[3:world_k]

                cols = st.columns(3,gap="medium")

                for i,m in enumerate(visible):

                    with cols[i%3]:
                        render_match_card(m)

                if extra:

                    with st.expander(f"More logo-like matches ({len(extra)})"):

                        more_cols = st.columns(3,gap="medium")

                        for i,m in enumerate(extra):

                            with more_cols[i%3]:
                                render_match_card(m)

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

            sem = semantic_mismatch(f,kw)

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
                    extra_category=(extra_cat if extra_cat!="-" else None),
                    similarity_rows=[],
                    cliches=cliches,
                    trend=tr,
                    semantic=sem,
                )

                st.download_button(
                    "Click to download",
                    data=pdf_bytes,
                    file_name="vcd_risk_sheet.pdf",
                    mime="application/pdf"
                )

        except Exception as e:

            st.error(f"Error while analyzing: {e}")
            st.stop()
