import os
import streamlit as st

from vcd_utils import CATEGORIES, load_image, extract_features, load_reference_features, similarity_to_set, detect_cliches, trend_risk, semantic_mismatch, world_scan


from pdf_utils import make_risk_sheet_pdf

APP_TITLE = "Visual Cliché Detector (MVP)"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(
    "Diagnostic prototype: similarity + cliché signals + trend risk + optional semantic tension. "
    "(Heuristic; not legal/trademark search.)"
)

data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)

colL, colR = st.columns([1, 2], gap="large")

with colL:
    st.subheader("Input")
    up = st.file_uploader("Upload logo mark (SVG/PNG/JPG)", type=["svg", "png", "jpg", "jpeg", "webp"])

    category = st.selectbox("Category context", options=CATEGORIES, index=0)
    extra_cat = st.selectbox("Optional second context", options=["-"] + CATEGORIES, index=0)
    keywords = st.text_input("Positioning keywords (comma-separated, optional)", placeholder="human, warm, bold")

    world_on = st.checkbox("World scan (web)", value=False)
    world_k = st.slider("How many results?", 3, 15, 8) if world_on else 0

    st.markdown("---")
    st.subheader("Reference set")
    st.write("Put reference logos into: `data/<category>/` to power similarity.")
    run = st.button("Analyze", type="primary", disabled=(up is None))


with colR:
    st.subheader("Report")

    if not up:
        st.info("Upload a mark to start. If SVG fails on cloud, upload PNG/JPG (export from Illustrator/Corel/etc.).")
    elif run:
        try:
            img = load_image(up.getvalue(), up.name)
            st.image(img, caption="Uploaded mark", width=260)

            f = extract_features(img)

            # --- World scan (web) ---
            world_results = []
            if world_on:
                with st.spinner("World scan: searching the web…"):
                    world_results = world_scan(img, max_results=world_k)

            # --- Similarity vs local reference sets ---
            refs1, ref_names1 = load_reference_features(data_dir, category)
            sim1, best_i1 = similarity_to_set(f, refs1)
            best_name1 = ref_names1[best_i1] if (best_i1 is not None and best_i1 < len(ref_names1)) else None

            sims = [(category, sim1, best_name1)]

            if extra_cat != "-":
                refs2, ref_names2 = load_reference_features(data_dir, extra_cat)
                sim2, best_i2 = similarity_to_set(f, refs2)
                best_name2 = ref_names2[best_i2] if (best_i2 is not None and best_i2 < len(ref_names2)) else None
                sims.append((extra_cat, sim2, best_name2))

            st.markdown("## Similarity")
            for cat, sim, best in sims:
                st.metric(label=f"{cat}", value=f"{sim:.1f}%")
                st.caption(f"Best match: {best}" if best else "No references found yet. Add files to data/...")
                            def proximity_label(score):
                try:
                    s = float(score)
                except Exception:
                    return "Unknown"

                if s <= 0:
                    return "Unknown"

                if s <= 1.0:
                    if s >= 0.85:
                        return "Critical"
                    if s >= 0.70:
                        return "High"
                    if s >= 0.55:
                        return "Moderate"
                    return "Low"

                if s >= 85:
                    return "Critical"
                if s >= 70:
                    return "High"
                if s >= 55:
                    return "Moderate"
                return "Low"


                        # --- Show World scan results ---
            if world_on:
                st.markdown("## 🌐 World scan (web)")

                with st.expander("debug: world_results raw"):
                    st.write(world_results)
                if not world_results:
                    st.info("No web matches found (or API not configured / Cloudinary upload failed).")
                else:
                    cols = st.columns(3, gap="medium")

                    for i, r in enumerate(world_results[:3]):
                        with cols[i % 3]:
                            thumb = r.get("thumbnail", "")
                            title = r.get("title", "") or "Result"
                            score = r.get("score", 0)
                            label = proximity_label(score)

                            if thumb:
                                st.image(thumb, use_container_width=True)

                            st.caption(title)
                            st.caption(f"Proximity: **{label}**")

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
                    similarity_rows=[{"category": c, "similarity": s, "best_match": b} for (c, s, b) in sims],
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
