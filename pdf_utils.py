
from __future__ import annotations
import io
from typing import Dict, List, Optional
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

def make_risk_sheet_pdf(logo_img: Image.Image, category: str, similarity: float, best_match_name: Optional[str],
                        cliches: List[Dict[str, str]], trend: Dict[str, str], semantic: Optional[Dict[str, str]]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    margin = 16 * mm
    x = margin
    y = h - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "Visual Cliché Detector — Risk Sheet")
    y -= 10 * mm

    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"Category: {category}")
    y -= 5 * mm
    c.drawString(x, y, f"Similarity (best match): {similarity:.1f}%")
    y -= 5 * mm
    if best_match_name:
        c.drawString(x, y, f"Best match file: {best_match_name}")
        y -= 5 * mm

    thumb = logo_img.convert("RGBA").resize((256, 256))
    img_buf = io.BytesIO()
    thumb.save(img_buf, format="PNG")
    img_buf.seek(0)

    y -= 3 * mm
    c.drawInlineImage(img_buf, x, y - 256, width=64*mm, height=64*mm)

    right_x = x + 70*mm
    block_y = y

    c.setFont("Helvetica-Bold", 12)
    c.drawString(right_x, block_y, "Trend risk")
    c.setFont("Helvetica", 10)
    block_y -= 5*mm
    c.drawString(right_x, block_y, f"Status: {trend.get('status','')}")
    block_y -= 5*mm
    c.drawString(right_x, block_y, (trend.get('note','')[:120]))

    block_y -= 10*mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(right_x, block_y, "Cliché signals")
    c.setFont("Helvetica", 10)
    block_y -= 5*mm
    if not cliches:
        c.drawString(right_x, block_y, "No strong signals detected (heuristic).")
        block_y -= 5*mm
    else:
        for s in cliches[:5]:
            c.setFont("Helvetica-Bold", 10)
            c.drawString(right_x, block_y, f"• {s.get('title','')}")
            block_y -= 4*mm
            c.setFont("Helvetica", 9)
            c.drawString(right_x + 3*mm, block_y, s.get("desc","")[:120])
            block_y -= 5*mm

    y2 = y - 74*mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y2, "Semantic check")
    y2 -= 5*mm
    c.setFont("Helvetica", 10)
    if semantic:
        c.drawString(x, y2, f"Status: {semantic.get('status','')}")
        y2 -= 5*mm
        c.drawString(x, y2, semantic.get("note","")[:180])
    else:
        c.drawString(x, y2, "Not provided.")

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(x, 12*mm, "Disclaimer: heuristic diagnostic; not a trademark search; not legal advice.")
    c.showPage()
    c.save()
    return buf.getvalue()
