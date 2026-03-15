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
            --surface-2: #f9fbff;
            --line: #dbe5f0;
            --text: #0f172a;
            --muted: #5b6472;
            --blue: #2563eb;
            --blue-2: #1d4ed8;
            --blue-soft: #eef4ff;
            --green: #15803d;
            --green-soft: #eefbf3;
            --red: #dc2626;
            --red-soft: #fff2f2;
            --amber: #b45309;
            --amber-soft: #fff8eb;
            --shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            --radius: 18px;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: linear-gradient(180deg, #f7f9fc 0%, #f3f6fb 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1360px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        h1 {
            font-weight: 800 !important;
            font-size: 2.25rem !important;
            margin-bottom: 0.35rem !important;
        }

        h2 {
            font-weight: 700 !important;
            margin-top: 0.3rem !important;
        }

        h3 {
            font-weight: 700 !important;
        }

        p, li, div[data-testid="stMarkdownContainer"] {
            color: var(--text);
        }

        div[data-testid="stHorizontalBlock"] {
            gap: 1.5rem;
        }

        div[data-testid="stFileUploader"] > label,
        div[data-testid="stButton"] > label {
            display: none !important;
        }

        div[data-testid="stFileUploaderDropzone"] {
            background: rgba(255,255,255,0.92);
            border: 1.5px dashed #bfd0e6 !important;
            border-radius: 24px !important;
            min-height: 210px !important;
            padding: 1.35rem 1.25rem !important;
            box-shadow: var(--shadow);
            transition: all 0.18s ease;
        }

        div[data-testid="stFileUploaderDropzone"]:hover {
            border-color: #7aa6ff !important;
            background: #fbfdff;
        }

        div[data-testid="stFileUploaderDropzone"] small {
            color: var(--muted) !important;
        }

        div[data-testid="stFileUploaderDropzone"] button {
            border-radius: 999px !important;
            min-height: 48px !important;
            padding: 0.75rem 1.1rem !important;
            border: 1px solid #c8d7ea !important;
            background: #ffffff !important;
            color: var(--text) !important;
            font-weight: 600 !important;
        }

        div[data-testid="stButton"] > button {
            width: 100%;
            min-height: 56px;
            border-radius: 16px;
            border: 0 !important;
            background: linear-gradient(180deg, #2f6df6 0%, #245ce0 100%) !important;
            color: white !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.24);
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }

        div[data-testid="stButton"] > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(37, 99, 235, 0.28);
        }

        div[data-testid="stButton"] > button:disabled {
            opacity: 0.45;
            box-shadow: none;
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
            margin-bottom: 0.15rem;
        }

        .lr-brand-fallback {
            font-size: 2.15rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #0f172a;
            line-height: 1;
        }

        .lr-brand-logo {
            max-height: 40px;
            width: auto;
            display: block;
        }

        .lr-subtitle {
            color: var(--muted);
            font-size: 0.98rem;
            margin-bottom: 1rem;
        }

        .lr-section-gap {
            height: 12px;
        }

        .lr-scan-pending {
            background: linear-gradient(180deg, #f7faff 0%, #eef4ff 100%);
            border: 1px solid #d6e4ff;
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem 1rem;
            margin: 0.25rem 0 1rem 0;
            box-shadow: var(--shadow);
        }

        .lr-scan-pending-title {
            font-weight: 700;
            color: #17356f;
            margin-bottom: 0.2rem;
        }

        .lr-scan-pending-note {
            color: #4f5d75;
            font-size: 0.92rem;
        }

        .lr-signal-card {
            background: rgba(255,255,255,0.92);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.7rem;
            box-shadow: var(--shadow);
        }

        .lr-signal-title {
            font-weight: 700;
            margin-bottom: 0.22rem;
            color: var(--text);
        }

        .lr-signal-desc {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.5;
            margin-bottom: 0.45rem;
        }

        .lr-signal-meta {
            color: #6b7280;
            font-size: 0.82rem;
        }

        .lr-trend-card {
            background: rgba(255,255,255,0.92);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: var(--shadow);
            margin-bottom: 0.5rem;
        }

        .lr-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.34rem 0.72rem;
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 0.65rem;
            border: 1px solid transparent;
        }

        .lr-badge-neutral {
            background: #eef2f7;
            color: #334155;
            border-color: #d9e2ec;
        }

        .lr-badge-green {
            background: var(--green-soft);
            color: var(--green);
            border-color: #cbeed6;
        }

        .lr-badge-red {
            background: var(--red-soft);
            color: var(--red);
            border-color: #ffd6d6;
        }

        .lr-badge-amber {
            background: var(--amber-soft);
            color: var(--amber);
            border-color: #f7dfae;
        }

        .lr-match-card {
            background: rgba(255,255,255,0.95);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.9rem;
            box-shadow: var(--shadow);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            gap: 0.7rem;
            min-height: 345px;
        }

        .lr-match-thumb {
            width: 100%;
            height: 170px;
            object-fit: contain;
            border-radius: 14px;
            background: #f8fafc;
            border: 1px solid #ecf1f7;
            padding: 10px;
        }

        .lr-match-title {
            font-size: 0.95rem;
            font-weight: 700;
            line-height: 1.35;
            color: var(--text);
            min-height: 2.6em;
            overflow: hidden;
        }

        .lr-match-meta {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            flex-wrap: wrap;
        }

        .lr-match-domain {
            color: var(--muted);
            font-size: 0.82rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 100%;
        }

        .lr-source-link {
            margin-top: auto;
            font-size: 0.88rem;
            font-weight: 600;
        }

        .lr-source-link a {
            color: var(--blue-2);
            text-decoration: none;
        }

        .lr-source-link a:hover {
            text-decoration: underline;
        }

        .lr-divider {
            margin-top: 1.6rem;
            padding-top: 1rem;
            border-top: 1px solid #dbe5f0;
        }

        .lr-disclaimer {
            color: #6b7280;
            font-size: 0.82rem;
            line-height: 1.55;
            max-width: 900px;
            padding-top: 0.2rem;
        }

        .stExpander {
            border: 1px solid var(--line) !important;
            border-radius: 16px !important;
            background: rgba(255,255,255,0.88) !important;
            box-shadow: none !important;
        }

        .stExpander details summary {
            font-weight: 600 !important;
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


def proximity_badge_class(label: str) -> str:
    if label == "High":
        return "lr-badge-red"
    if label == "Medium":
        return "lr-badge-amber"
    if label == "Low":
        return "lr-badge-green"
    return "lr-badge-neutral"


def trend_badge_class(status: str) -> str:
    s = (status or "").lower()
    if s in {"saturated", "peak"}:
        return "lr-badge-red"
    if s in {"rising"}:
        return "lr-badge-amber"
    return "lr-badge-neutral"


def domain_of(url: str):
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def title_is_noise(title: str, link: str):
    text = f"{title} {link}".lower()

    noisy_terms = [
        "youtube", "youtu.be", "reddit", "pinterest", "tiktok", "facebook",
        "instagram", "shutterstock", "freepik", "clipart", "meme",
        "tutorial", "how to", "exercise", "lyrics", "song", "video",
        "drawing", "sketch", "doodle", "cartoon", "illustration", "wallpaper",
        "diagram", "schematic", "process", "step by step",
    ]

    noisy_domains = [
        "youtube.com", "youtu.be", "reddit.com", "pinterest.com",
        "tiktok.com", "facebook.com", "instagram.com",
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
        "logo", "brand", "branding", "identity", "company", "official",
        "inc", "ltd", "llc", "group", "studio", "agency", "services",
        "hospitality", "design", "projects", "portfolio", "case study",
        "help center", "about us",
    ]

    brand_domains = [
        "1000logos", "behance", "dribbble", "brandsoftheworld",
        "logowik", "logos-world", "crunchbase", "linkedin",
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

    is_noise = title_is_noise(title, link)
    is_logo_like = title_is_logo_like(title, link)
    is_relevant = (not is_noise) and is_logo_like

    return {
        "thumb": thumb,
        "title": title,
        "link": link,
        "label": label,
        "sim_pct": sim_pct,
        "is_relevant": is_relevant,
        "is_noise": is_noise,
        "is_logo_like": is_logo_like,
    }


def label_rank(label: str) -> int:
    order = {
        "High": 0,
        "Medium": 1,
        "Low": 2,
        "Unknown": 3,
    }
    return order.get(label, 99)


def stable_sort_matches(matches):
    def sort_key(m):
        sim = m["sim_pct"] if m["sim_pct"] is not None else -1
        return (
            0 if m["is_relevant"] else 1,
            label_rank(m["label"]),
            -sim,
            m["title"].lower(),
        )

    return sorted(matches, key=sort_key)


def warning_state(matches):
    relevant = [m for m in matches if m["is_relevant"]][:WARNING_POOL]

    high_count = sum(1 for m in relevant if m["label"] == "High")
    medium_count = sum(1 for m in relevant if m["label"] == "Medium")

    if high_count >= 2:
        return "high"
    if high_count >= 1 and (high_count + medium_count) >= 2:
        return "high"
    if high_count >= 1:
        return "mixed"
    if medium_count >= 2:
        return "mixed"
    return "safe"


def render_logo_brand():
    possible_paths = [
        os.path.join("assets", "logoradar-logo.svg"),
        os.path.join("assets", "logoradar-logo.png"),
        os.path.join("assets", "logoradar-logo.jpg"),
    ]

    found = None
    for p in possible_paths:
        if os.path.exists(p):
            found = p
            break

    if found:
        ext = os.path.splitext(found)[1].lower()
        mime = "image/svg+xml" if ext == ".svg" else f"image/{ext.replace('.', '')}"
        with open(found, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(
            f"""
            <div class="lr-brand-wrap">
                <img class="lr-brand-logo" src="data:{mime};base64,{encoded}" alt="LogoRadar logo">
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="lr-brand-wrap">
                <div class="lr-brand-fallback">LogoRadar</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_match_card(match):
    thumb = match.get("thumb") or ""
    title = match.get("title") or "Result"
    link = match.get("link") or ""
    label = match.get("label") or "Unknown"
    domain = domain_of(link)
    badge_class = proximity_badge_class(label)

    thumb_html = ""
    if thumb:
        thumb_html = f'<img class="lr-match-thumb" src="{thumb}" alt="{title}">'
    else:
        thumb_html = '<div class="lr-match-thumb"></div>'

    source_html = ""
    if link:
        source_html = f'<div class="lr-source-link"><a href="{link}" target="_blank">Open source</a></div>'

    st.markdown(
        f"""
        <div class="lr-match-card">
            {thumb_html}
            <div class="lr-match-title">{title}</div>
            <div class="lr-match-meta">
                <span class="lr-badge {badge_class}">{label}</span>
            </div>
            <div class="lr-match-domain">{domain}</div>
            {source_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_card(signal):
    st.markdown(
        f"""
        <div class="lr-signal-card">
            <div class="lr-signal-title">{signal['title']}</div>
            <div class="lr-signal-desc">{signal['desc']}</div>
            <div class="lr-signal-meta">Common in: {signal['common_in']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trend_card(tr):
    badge_class = trend_badge_class(tr["status"])
    st.markdown(
        f"""
        <div class="lr-trend-card">
            <div class="lr-badge {badge_class}">{tr['status']}</div>
            <div class="lr-signal-desc" style="margin-bottom:0;">{tr['note']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_ui_css()

st.set_page_config(page_title=APP_TITLE, layout="wide")

render_logo_brand()
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
        help="Upload a logo mark to scan for visual overlap, cliché signals and trend risk.",
    )

    st.markdown('<div class="lr-section-gap"></div>', unsafe_allow_html=True)

    run = st.button("Analyze", type="primary", disabled=(up is None), use_container_width=True)

with colR:
    st.subheader("Report")

    if not up:
        st.info("Upload a mark to start.")

    elif run:
        try:
            file_bytes = up.getvalue()
            file_name = up.name

            img = load_image(file_bytes, file_name)
            f = extract_features(img)

            st.image(img, caption="Uploaded mark", width=280)

            scan_box = st.empty()

            with scan_box.container():
                st.markdown(
                    """
                    <div class="lr-scan-pending">
                        <div class="lr-scan-pending-title">Web scan in progress...</div>
                        <div class="lr-scan-pending-note">Searching the web for similar marks.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.subheader("Cliché signals")

            cliches = detect_cliches(f)
            if not cliches:
                st.success("No strong cliché signals detected.")
            else:
                for s in cliches:
                    render_signal_card(s)

            st.subheader("Trend risk")
            tr = trend_risk(f)
            render_trend_card(tr)

            world_results = cached_world_scan(file_bytes, file_name)

            match_data = [build_match_data(r, f) for r in world_results]
            match_data = stable_sort_matches(match_data)

            relevant_matches = [m for m in match_data if m["is_relevant"]]
            state = warning_state(match_data)

            with scan_box.container():
                st.subheader("Early warning")

                if state == "high":
                    st.error("Designer, be careful. Strong logo-like visual overlap was found online.")
                elif state == "mixed":
                    st.warning("Some potentially relevant logo-like matches were found.")
                else:
                    st.success("Looks safe. No meaningful logo-like matches found online.")

                st.subheader("World scan")

                if not relevant_matches:
                    st.info("No logo-like matches found.")
                else:
                    visible = relevant_matches[:VISIBLE_MATCHES]
                    extra = relevant_matches[VISIBLE_MATCHES:]

                    cols = st.columns(3, gap="medium")
                    for i, m in enumerate(visible):
                        with cols[i % 3]:
                            render_match_card(m)

                    if extra:
                        with st.expander(f"More matches ({len(extra)})", expanded=False):
                            more_cols = st.columns(3, gap="medium")
                            for i, m in enumerate(extra):
                                with more_cols[i % 3]:
                                    render_match_card(m)

        except Exception as e:
            st.error(f"Error while analyzing: {e}")
            st.stop()

st.markdown('<div class="lr-divider"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="lr-disclaimer">
        Disclaimer: This tool supports creative exploration and early visual checks.
        It is not a legal trademark search and does not replace professional trademark clearance.
    </div>
    """,
    unsafe_allow_html=True,
)
