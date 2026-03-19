import streamlit as st
from PIL import Image
from transformers import pipeline
import torch
import plotly.graph_objects as go
import cv2
import numpy as np

st.set_page_config(layout="wide")

ETHNICITY_MAP = {
    "European": {
        "color": "#4A90D9",
        "countries": [
            "DEU","FRA","GBR","ITA","ESP","POL","ROU","NLD","BEL","SWE",
            "CZE","GRC","PRT","HUN","AUT","CHE","BGR","DNK","FIN","SVK",
            "NOR","IRL","HRV","BIH","ALB","LTU","SVN","LVA","EST","MKD",
            "LUX","MNE","MLT","ISL","RUS","UKR","BLR","MDA","SRB",
            "AND","LIE","SMR","VAT","MCO","GEO","ARM","AZE",
        ],
    },
    "Latino Hispanic": {
        "color": "#E8A838",
        "countries": ["MEX","COL","ARG","PER","VEN","CHL","ECU","GTM","CUB","BOL",
                      "DOM","HND","PRY","SLV","NIC","CRI","PAN","URY","BRA"],
    },
    "South Asian": {
        "color": "#7ED321",
        "countries": ["IND","PAK","BGD","LKA","NPL","BTN","MDV"],
    },
    "East Asian": {
        "color": "#F5A0C0",
        "countries": ["CHN","JPN","KOR","MNG","TWN","HKG","MAC","VNM","THA","IDN",
                      "PHL","MYS","MMR","KHM","LAO","SGP","BRN","TLS"],
    },
    "African": {
        "color": "#C0703A",
        "countries": ["NGA","ETH","COD","TZA","KEN","ZAF","UGA","GHA","MOZ","AGO",
                      "MDG","CMR","CIV","SEN","ZMB","ZWE","MLI","BFA","NER","SOM",
                      "EGY","DZA","MAR","LBY","TUN","SDN"],
    },
    "Middle Eastern": {
        "color": "#9B59B6",
        "countries": ["SAU","IRN","IRQ","SYR","JOR","LBN","ISR","YEM","OMN","ARE",
                      "KWT","QAT","BHR","TUR"],
    },
}

LABEL_MAPPING = {
    "caucasian": "European", "white": "European", "european": "European",
    "hispanic": "Latino Hispanic", "latino": "Latino Hispanic", "latino hispanic": "Latino Hispanic",
    "indian": "South Asian", "south asian": "South Asian",
    "asian": "East Asian", "east asian": "East Asian", "southeast asian": "East Asian",
    "black": "African", "african": "African", "african american": "African",
    "middle eastern": "Middle Eastern", "arab": "Middle Eastern",
}

def match_label(raw_label: str):
    raw = raw_label.lower().replace("_", " ")
    for key, val in LABEL_MAPPING.items():
        if key in raw:
            return val
    return None

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

@st.cache_resource
def load_classifier():
    return pipeline(
        "image-classification",
        model="cledoux42/Ethnicity_Test_v003",
        device=0 if torch.cuda.is_available() else -1,
    )

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

classifier   = load_classifier()
face_cascade = load_face_cascade()

def draw_face_annotations(pil_image: Image.Image):
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray    = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE,
    )

    NEON  = (255, 225, 0)   # cyan in BGR
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    for (x, y, w, h) in faces:
        # semi-transparent fill
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), NEON, -1)
        cv2.addWeighted(overlay, 0.10, img_bgr, 0.90, 0, img_bgr)

        # dashed border
        def dashed_rect(img, x, y, w, h, color, thickness=2, dash=16, gap=8):
            for (x1,y1),(x2,y2) in [
                ((x,y),(x+w,y)), ((x+w,y),(x+w,y+h)),
                ((x+w,y+h),(x,y+h)), ((x,y+h),(x,y))
            ]:
                dist = int(np.hypot(x2-x1, y2-y1))
                if dist == 0: continue
                dx, dy = (x2-x1)/dist, (y2-y1)/dist
                pos, on = 0, True
                while pos < dist:
                    seg = min(pos + (dash if on else gap), dist)
                    if on:
                        cv2.line(img,
                                 (int(x1+dx*pos), int(y1+dy*pos)),
                                 (int(x1+dx*seg), int(y1+dy*seg)),
                                 color, thickness, cv2.LINE_AA)
                    pos += dash if on else gap
                    on = not on

        dashed_rect(img_bgr, x, y, w, h, NEON, thickness=2)

        # corner brackets
        arm = max(20, w // 7)
        for (cx, cy), (sx, sy) in [
            ((x,   y  ),( 1, 1)), ((x+w, y  ),(-1, 1)),
            ((x,   y+h),( 1,-1)), ((x+w, y+h),(-1,-1)),
        ]:
            cv2.line(img_bgr, (cx, cy), (cx+sx*arm, cy),     WHITE, 3, cv2.LINE_AA)
            cv2.line(img_bgr, (cx, cy), (cx, cy+sy*arm),     WHITE, 3, cv2.LINE_AA)
            cv2.circle(img_bgr, (cx, cy), 5, NEON, -1, cv2.LINE_AA)

        # label banner
        banner_h = 28
        by = max(0, y - banner_h)
        cv2.rectangle(img_bgr, (x, by), (x+w, y), NEON, -1)
        label = "FACE DETECTED"
        font, fscale, fthick = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
        (tw, th), _ = cv2.getTextSize(label, font, fscale, fthick)
        cv2.putText(img_bgr, label,
                    (x + (w-tw)//2, by + (banner_h+th)//2),
                    font, fscale, BLACK, fthick, cv2.LINE_AA)

    annotated = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated), len(faces) > 0

# ── CSS: make camera widget fill width ───────────────────────────────────────
st.markdown("""
<style>
[data-testid="stCameraInput"] { width: 100%; }
[data-testid="stCameraInput"] > div { width: 100%; }
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img   { width: 100% !important; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

cam_photo = st.camera_input("", label_visibility="collapsed")

if cam_photo:
    col1, col2 = st.columns([1, 1])
    with col1:
        raw_pil = Image.open(cam_photo).convert("RGB")
        annotated_pil, face_found = draw_face_annotations(raw_pil)

        if face_found:
            st.success("Face detected — running ethnicity analysis…")
        else:
            st.warning("No face detected — try better lighting or move closer")

        st.image(annotated_pil, use_container_width=True)
        img_input = cam_photo

        if img_input:
            image = Image.open(img_input).convert("RGB")

        if img_input is not cam_photo:
            st.image(image)

        with st.spinner("Analysing…"):
            results = classifier(image, top_k=10)

    with col2:
        st.markdown("### Results")
        for r in results:
            label   = r["label"].replace("_"," ").title()
            score   = r["score"] * 100
            matched = match_label(r["label"])
            color   = ETHNICITY_MAP[matched]["color"] if matched else "#888888"
            bar_w   = int(score * 1.8)
            st.markdown(
                f'<div style="display:flex;align-items:center;margin-bottom:6px;gap:8px;">'
                f'<span style="width:12px;height:12px;border-radius:50%;background:{color};'
                f'display:inline-block;flex-shrink:0;"></span>'
                f'<span style="width:140px;font-size:13px;"><b>{label}</b></span>'
                f'<div style="background:#eee;border-radius:4px;height:10px;width:180px;">'
                f'<div style="background:{color};width:{bar_w}px;height:10px;border-radius:4px;"></div>'
                f'</div>'
                f'<span style="font-size:12px;color:#555;">{score:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        fig = go.Figure()

        for r in results:
            matched = match_label(r["label"])
            if not matched:
                continue
            score         = r["score"] * 100
            info          = ETHNICITY_MAP[matched]
            label_display = r["label"].replace("_"," ").title()

            fig.add_trace(go.Choropleth(
                locations=info["countries"],
                z=[score] * len(info["countries"]),
                zmin=0,
                zmax=100,
                # Low end = visible tint (0.25 alpha), high end = full color
                colorscale=[[0, hex_to_rgba(info["color"], 0.25)], [1, info["color"]]],
                showscale=False,
                name=label_display,
                hovertemplate=f"<b>%{{location}}</b><br>{label_display}: {score:.1f}%<extra></extra>",
                marker_line_color="#cccccc",
                marker_line_width=0.5,
            ))

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="white",
            geo=dict(
                bgcolor="white", showframe=False, showcoastlines=False,
                showland=True, landcolor="#eeeeee",
                showocean=True, oceancolor="#ddeeff",
                showcountries=True, countrycolor="#cccccc", showlakes=False,
            ),
            height=440,
        )

        st.plotly_chart(fig, use_container_width=True)