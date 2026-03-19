import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
import torch
import plotly.graph_objects as go
import cv2
import numpy as np
import io

st.set_page_config(layout="wide")

ETHNICITY_MAP = {
    "Europeo": {
        "color": "#4A90D9",
        "countries": [
            "DEU","FRA","GBR","ITA","ESP","POL","ROU","NLD","BEL","SWE",
            "CZE","GRC","PRT","HUN","AUT","CHE","BGR","DNK","FIN","SVK",
            "NOR","IRL","HRV","BIH","ALB","LTU","SVN","LVA","EST","MKD",
            "LUX","MNE","MLT","ISL","RUS","UKR","BLR","MDA","SRB",
            "AND","LIE","SMR","VAT","MCO","GEO","ARM","AZE",
        ],
    },
    "Latino Ispanico": {
        "color": "#E8A838",
        "countries": ["MEX","COL","ARG","PER","VEN","CHL","ECU","GTM","CUB","BOL",
                      "DOM","HND","PRY","SLV","NIC","CRI","PAN","URY","BRA"],
    },
    "Asiatico del Sud": {
        "color": "#7ED321",
        "countries": ["IND","PAK","BGD","LKA","NPL","BTN","MDV"],
    },
    "Asiatico Orientale": {
        "color": "#F5A0C0",
        "countries": ["CHN","JPN","KOR","MNG","TWN","HKG","MAC","VNM","THA","IDN",
                      "PHL","MYS","MMR","KHM","LAO","SGP","BRN","TLS"],
    },
    "Africano": {
        "color": "#C0703A",
        "countries": [
            "NGA","ETH","COD","TZA","KEN","ZAF","UGA","GHA","MOZ","AGO",
            "MDG","CMR","CIV","SEN","ZMB","ZWE","MLI","BFA","NER","SOM",
            "EGY","DZA","MAR","LBY","TUN","SDN",
            "BEN","BWA","BDI","CPV","CAF","TCD","COM","COG","DJI","GNQ",
            "ERI","SWZ","GAB","GMB","GIN","GNB","LSO","LBR","MWI","MRT",
            "MUS","NAM","RWA","STP","SLE","SSD","TGO","ESH",
        ],
    },
    "Mediorientale": {
        "color": "#9B59B6",
        "countries": ["SAU","IRN","IRQ","SYR","JOR","LBN","ISR","YEM","OMN","ARE",
                      "KWT","QAT","BHR","TUR"],
    },
}

LABEL_MAPPING = {
    "caucasian": "Europeo", "white": "Europeo", "european": "Europeo",
    "hispanic": "Latino Ispanico", "latino": "Latino Ispanico", "latino hispanic": "Latino Ispanico",
    "indian": "Asiatico del Sud", "south asian": "Asiatico del Sud",
    "asian": "Asiatico Orientale", "east asian": "Asiatico Orientale", "southeast asian": "Asiatico Orientale",
    "black": "Africano", "african": "Africano", "african american": "Africano",
    "middle eastern": "Mediorientale", "arab": "Mediorientale",
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

def draw_base_frame(img_bgr, faces, scan_x_offset=None):
    """Draw face box + optional scan bar onto img_bgr (in-place copy). Returns annotated BGR image."""
    out = img_bgr.copy()
    NEON  = (0, 225, 255)   # BGR
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    for (x, y, w, h) in faces:
        # Semi-transparent fill
        overlay = out.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), NEON, -1)
        cv2.addWeighted(overlay, 0.08, out, 0.92, 0, out)

        # ── Scanning bar ──────────────────────────────────────────────────
        if scan_x_offset is not None:
            bx = x + scan_x_offset
            bar_w = max(6, w // 10)
            # clamp to box
            bx = max(x, min(bx, x + w - bar_w))
            bar_overlay = out.copy()
            cv2.rectangle(bar_overlay, (bx, y+2), (bx + bar_w, y+h-2), NEON, -1)
            cv2.addWeighted(bar_overlay, 0.55, out, 0.45, 0, out)
            # bright leading edge
            cv2.line(out, (bx + bar_w, y+2), (bx + bar_w, y+h-2), WHITE, 2, cv2.LINE_AA)

        # ── Dashed border ─────────────────────────────────────────────────
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

        dashed_rect(out, x, y, w, h, NEON, thickness=2)

        # ── Corner brackets ───────────────────────────────────────────────
        arm = max(20, w // 7)
        for (cx, cy), (sx, sy) in [
            ((x,   y  ),( 1, 1)), ((x+w, y  ),(-1, 1)),
            ((x,   y+h),( 1,-1)), ((x+w, y+h),(-1,-1)),
        ]:
            cv2.line(out, (cx, cy), (cx+sx*arm, cy),  WHITE, 3, cv2.LINE_AA)
            cv2.line(out, (cx, cy), (cx, cy+sy*arm),  WHITE, 3, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 5, NEON, -1, cv2.LINE_AA)

        # ── Label banner ──────────────────────────────────────────────────
        banner_h = 28
        by = max(0, y - banner_h)
        cv2.rectangle(out, (x, by), (x+w, y), NEON, -1)
        label = ("ANALIZZO...") if scan_x_offset is not None else "Volto rilevato"
        font, fscale, fthick = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
        (tw, th), _ = cv2.getTextSize(label, font, fscale, fthick)
        cv2.putText(out, label,
                    (x + (w-tw)//2, by + (banner_h+th)//2),
                    font, fscale, BLACK, fthick, cv2.LINE_AA)

    return out

def create_scanning_gif(pil_image: Image.Image, faces) -> bytes:
    """
    Build an animated GIF: scanning bar sweeps left→right→left inside each face box.
    Returns raw GIF bytes.
    """
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Resize for speed if very large
    max_dim = 640
    h0, w0 = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h0, w0))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w0*scale), int(h0*scale)))
        faces = [(int(x*scale), int(y*scale), int(w*scale), int(h*scale)) for (x,y,w,h) in faces]

    N_FRAMES  = 15          # frames per sweep direction
    LOOP_REPS = 4           # number of full back-and-forth cycles
    FRAME_DUR = 50          # ms per frame  (≈16 fps)

    pil_frames = []

    for _ in range(LOOP_REPS):
        # left → right
        for i in range(N_FRAMES):
            t = i / (N_FRAMES - 1)          # 0 → 1
            frames_bgr = []
            for (x, y, w, h) in faces:
                offset = int(t * w)
            annotated = draw_base_frame(img_bgr, faces, scan_x_offset=offset)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))

        # right → left
        for i in range(N_FRAMES):
            t = 1.0 - i / (N_FRAMES - 1)
            frames_bgr = []
            for (x, y, w, h) in faces:
                offset = int(t * w)
            annotated = draw_base_frame(img_bgr, faces, scan_x_offset=offset)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))

    buf = io.BytesIO()
    pil_frames[0].save(
        buf, format="GIF",
        save_all=True, append_images=pil_frames[1:],
        loop=0, duration=FRAME_DUR, optimize=False,
    )
    return buf.getvalue()

def draw_face_annotations(pil_image: Image.Image):
    """Static version — detects at most ONE face."""
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray    = cv2.equalizeHist(gray)
    all_faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE,
    )
    # Keep only the largest face by area
    faces = []
    if len(all_faces) > 0:
        largest = max(all_faces, key=lambda f: f[2] * f[3])
        faces = [largest]

    annotated = draw_base_frame(img_bgr, faces, scan_x_offset=None)
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), len(faces) > 0, faces

left, center, right = st.columns([1,1,1])

with center:
    cam_photo = st.camera_input("main_camera_input", label_visibility="collapsed",
                                width=600)

# Reset state when a new photo is taken
if cam_photo:
    photo_id = hash(cam_photo.getvalue())
    if st.session_state.get("photo_id") != photo_id:
        st.session_state["photo_id"]  = photo_id
        st.session_state["phase"]     = "scanning"   # scanning → done
        st.session_state["results"]   = None
        st.session_state["annotated"] = None
        st.session_state["gif"]       = None
        st.session_state["faces"]     = None
        st.session_state["raw_pil"]   = None

if cam_photo and st.session_state.get("phase"):
    phase = st.session_state["phase"]

    # ── Pre-compute once ──────────────────────────────────────────────────
    if st.session_state["raw_pil"] is None:
        raw_pil = Image.open(cam_photo).convert("RGB")
        annotated_pil, face_found, faces = draw_face_annotations(raw_pil)
        st.session_state["raw_pil"]   = raw_pil
        st.session_state["annotated"] = annotated_pil
        st.session_state["face_found"] = face_found
        st.session_state["faces"]     = faces[:1]  # enforce single face

    raw_pil       = st.session_state["raw_pil"]
    annotated_pil = st.session_state["annotated"]
    face_found    = st.session_state["face_found"]
    faces         = st.session_state["faces"]

    col1, col2 = st.columns([1, 1])

    with col1:
        if not face_found:
            st.warning("Nessun volto rilevato — prova con una luce migliore o avvicinati alla fotocamera.")
            st.image(annotated_pil, use_container_width=True)

        elif phase == "scanning":
            st.info("Volto rilevato — analisi in corso…")

            # Build GIF once
            if st.session_state["gif"] is None:
                st.session_state["gif"] = create_scanning_gif(raw_pil, faces)

            st.image(st.session_state["gif"], use_container_width=True)

            # Run classifier
            with st.spinner("Analisi in corso…"):
                results = classifier(raw_pil, top_k=10)
            st.session_state["results"] = results
            st.session_state["phase"]   = "done"

            # Wait 3 seconds showing the GIF, then rerun to flip to results
            import time
            time.sleep(9)
            st.rerun()

        else:  # phase == "done"
            st.success("Analisi completata!")
            st.image(annotated_pil, use_container_width=True)

    with col2:
        if face_found and st.session_state.get("results"):
            results = st.session_state["results"]
            st.subheader("Risultati")
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
                label_display = matched  # Use Italian group name directly
                fig.add_trace(go.Choropleth(
                    locations=info["countries"],
                    z=[score] * len(info["countries"]),
                    zmin=0, zmax=100,
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