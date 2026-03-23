import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
import torch
import plotly.graph_objects as go
import cv2
import numpy as np
import io
import time

st.set_page_config(layout="wide", page_title="Analisi Etnia", page_icon="🔬")

#  CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0a0e1a;
    color: #e0e8ff;
}

/* ── Page background */
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0d1b3e 0%, #0a0e1a 60%);
    min-height: 100vh;
}

/* ── Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }

/* ── Phase stepper */
.phase-stepper {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin: 0 auto 2.5rem auto;
    max-width: 640px;
    font-family: 'Share Tech Mono', monospace;
}
.phase-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    flex: 1;
}
.phase-dot {
    width: 36px; height: 36px;
    border-radius: 50%;
    border: 2px solid #2a3a6a;
    background: #10182e;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700; color: #3a5a9a;
    transition: all .4s ease;
    position: relative;
    z-index: 2;
}
.phase-dot.active {
    border-color: #00e5ff;
    background: linear-gradient(135deg, #001f3f, #003366);
    color: #00e5ff;
    box-shadow: 0 0 16px #00e5ff66, 0 0 4px #00e5ff;
}
.phase-dot.done {
    border-color: #00ff9d;
    background: linear-gradient(135deg, #001a10, #003322);
    color: #00ff9d;
    box-shadow: 0 0 10px #00ff9d44;
}
.phase-label {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3a5a9a;
}
.phase-label.active { color: #00e5ff; }
.phase-label.done   { color: #00ff9d; }
.phase-connector {
    flex: 1;
    height: 2px;
    margin-bottom: 22px;
    background: #2a3a6a;
}
.phase-connector.done { background: linear-gradient(90deg, #00ff9d, #00e5ff); }

/* ── Section card */
.phase-card {
    background: linear-gradient(145deg, #111a32, #0d1424);
    border: 1px solid #1e2e50;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 40px #00000066;
}
.phase-card h2 {
    font-family: 'Share Tech Mono', monospace;
    color: #00e5ff;
    font-size: 1rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #1e2e50;
    padding-bottom: .75rem;
}

/* ── Camera widget override */
[data-testid="stCameraInput"] > div {
    border: 1px solid #1e2e50 !important;
    border-radius: 12px !important;
    overflow: hidden;
    background: #0a0e1a !important;
}

/* ── Buttons */
.stButton > button {
    background: linear-gradient(135deg, #003366, #001f3f);
    border: 1px solid #00e5ff;
    color: #00e5ff;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 2px;
    font-size: 0.85rem;
    padding: .65rem 2rem;
    border-radius: 6px;
    transition: all .3s;
}
.stButton > button:hover {
    background: #00e5ff;
    color: #0a0e1a;
    box-shadow: 0 0 20px #00e5ff66;
}

/* ── Info / warning / success banners */
.stAlert {
    border-radius: 8px;
    font-family: 'Share Tech Mono', monospace;
    font-size: .85rem;
    letter-spacing: 1px;
}

/* ── Results bar overrides */
.result-row {
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# Data
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

# Models
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

# CV helpers
def draw_base_frame(img_bgr, faces, scan_x_offset=None):
    out = img_bgr.copy()
    NEON  = (0, 225, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    for (x, y, w, h) in faces:
        overlay = out.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), NEON, -1)
        cv2.addWeighted(overlay, 0.08, out, 0.92, 0, out)

        if scan_x_offset is not None:
            bx    = x + scan_x_offset
            bar_w = max(6, w // 10)
            bx    = max(x, min(bx, x + w - bar_w))
            bar_overlay = out.copy()
            cv2.rectangle(bar_overlay, (bx, y+2), (bx + bar_w, y+h-2), NEON, -1)
            cv2.addWeighted(bar_overlay, 0.55, out, 0.45, 0, out)
            cv2.line(out, (bx + bar_w, y+2), (bx + bar_w, y+h-2), WHITE, 2, cv2.LINE_AA)

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

        arm = max(20, w // 7)
        for (cx, cy), (sx, sy) in [
            ((x,   y  ),( 1, 1)), ((x+w, y  ),(-1, 1)),
            ((x,   y+h),( 1,-1)), ((x+w, y+h),(-1,-1)),
        ]:
            cv2.line(out, (cx, cy), (cx+sx*arm, cy),  WHITE, 3, cv2.LINE_AA)
            cv2.line(out, (cx, cy), (cx, cy+sy*arm),  WHITE, 3, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 5, NEON, -1, cv2.LINE_AA)

        banner_h = 28
        by = max(0, y - banner_h)
        cv2.rectangle(out, (x, by), (x+w, y), NEON, -1)
        label = "ANALIZZO..." if scan_x_offset is not None else "Volto rilevato"
        font, fscale, fthick = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
        (tw, th), _ = cv2.getTextSize(label, font, fscale, fthick)
        cv2.putText(out, label,
                    (x + (w-tw)//2, by + (banner_h+th)//2),
                    font, fscale, BLACK, fthick, cv2.LINE_AA)

    return out

def create_scanning_gif(pil_image, faces):
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    max_dim = 640
    h0, w0  = img_bgr.shape[:2]
    scale   = min(1.0, max_dim / max(h0, w0))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w0*scale), int(h0*scale)))
        faces   = [(int(x*scale), int(y*scale), int(w*scale), int(h*scale)) for (x,y,w,h) in faces]

    N_FRAMES  = 15
    LOOP_REPS = 4
    FRAME_DUR = 50
    pil_frames = []

    for _ in range(LOOP_REPS):
        for i in range(N_FRAMES):
            t = i / (N_FRAMES - 1)
            for (x, y, w, h) in faces:
                offset = int(t * w)
            annotated  = draw_base_frame(img_bgr, faces, scan_x_offset=offset)
            pil_frames.append(Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)))
        for i in range(N_FRAMES):
            t = 1.0 - i / (N_FRAMES - 1)
            for (x, y, w, h) in faces:
                offset = int(t * w)
            annotated  = draw_base_frame(img_bgr, faces, scan_x_offset=offset)
            pil_frames.append(Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)))

    buf = io.BytesIO()
    pil_frames[0].save(buf, format="GIF", save_all=True, append_images=pil_frames[1:],
                       loop=0, duration=FRAME_DUR, optimize=False)
    return buf.getvalue()

def draw_face_annotations(pil_image):
    img_bgr  = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray     = cv2.equalizeHist(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    all_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(60,60), flags=cv2.CASCADE_SCALE_IMAGE)
    faces = []
    if len(all_faces) > 0:
        faces = [max(all_faces, key=lambda f: f[2]*f[3])]
    annotated = draw_base_frame(img_bgr, faces, scan_x_offset=None)
    return Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)), len(faces) > 0, faces


# ── Helper: render the phase stepper ──────────────────────────────────────────
def render_stepper(phase: str):
    steps = [
        ("01", "FOTO",     "photo"),
        ("02", "ANALISI",  "analysis"),
        ("03", "RISULTATI","results"),
    ]
    order = ["photo", "analysis", "results"]
    current_idx = order.index(phase)

    html = '<div class="phase-stepper">'
    for i, (num, lbl, key) in enumerate(steps):
        idx = order.index(key)
        if idx < current_idx:
            dot_cls = "phase-dot done"
            lbl_cls = "phase-label done"
            icon    = "✓"
        elif idx == current_idx:
            dot_cls = "phase-dot active"
            lbl_cls = "phase-label active"
            icon    = num
        else:
            dot_cls = "phase-dot"
            lbl_cls = "phase-label"
            icon    = num

        html += f'<div class="phase-step"><div class="{dot_cls}">{icon}</div><span class="{lbl_cls}">{lbl}</span></div>'

        if i < len(steps) - 1:
            conn_cls = "phase-connector done" if current_idx > idx else "phase-connector"
            html += f'<div class="{conn_cls}"></div>'

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def card(title: str):
    st.markdown(f'<div class="phase-card"><h2>{title}</h2>', unsafe_allow_html=True)

def card_end():
    st.markdown('</div>', unsafe_allow_html=True)


if "phase" not in st.session_state:
    st.session_state["phase"] = "photo"

# ── Page title ─────────────────────────────────────────────────────────────────
# st.markdown("""
# <div style="text-align:center; margin-bottom:2rem;">
#   <div style="font-family:'Share Tech Mono',monospace; font-size:.75rem;
#               letter-spacing:6px; color:#3a6a9a; margin-bottom:.4rem;">
#     SISTEMA DI ANALISI BIOMETRICA
#   </div>
#   <div style="font-family:'Rajdhani',sans-serif; font-weight:700;
#               font-size:2rem; color:#e0e8ff; letter-spacing:4px;">
#     RICONOSCIMENTO DELL'ETNIA
#   </div>
# </div>
# """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-bottom:2rem;">
  <div style="font-family:'Rajdhani',sans-serif; font-weight:700;
              font-size:2rem; color:#e0e8ff; letter-spacing:4px;">
    RICONOSCIMENTO DELL'ETNIA
  </div>
</div>
""", unsafe_allow_html=True)

phase = st.session_state["phase"]
render_stepper(phase)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — PHOTO
# ══════════════════════════════════════════════════════════════════════════════
if phase == "photo":
    # card("ACQUISISCI IMMAGINE")
    _, cam_col, _ = st.columns([1, 2, 1])
    with cam_col:
        cam_photo = st.camera_input("Scatta una foto", label_visibility="collapsed")
        st.markdown("""
        <div style="text-align:center; margin-top:.75rem;
                    font-family:'Share Tech Mono',monospace; font-size:.7rem;
                    color:#3a6a9a; letter-spacing:2px;">
            POSIZIONA IL VOLTO AL CENTRO CON BUONA ILLUMINAZIONE
        </div>
        """, unsafe_allow_html=True)
    card_end()

    if cam_photo:
        photo_id = hash(cam_photo.getvalue())
        if st.session_state.get("photo_id") != photo_id:
            st.session_state["photo_id"]   = photo_id
            st.session_state["results"]    = None
            st.session_state["annotated"]  = None
            st.session_state["gif"]        = None
            st.session_state["faces"]      = None
            st.session_state["raw_pil"]    = None
            st.session_state["face_found"] = None

        raw_pil = Image.open(cam_photo).convert("RGB")
        annotated_pil, face_found, faces = draw_face_annotations(raw_pil)
        st.session_state["raw_pil"]    = raw_pil
        st.session_state["annotated"]  = annotated_pil
        st.session_state["face_found"] = face_found
        st.session_state["faces"]      = faces[:1]

        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            if face_found:
                st.success("✔  Volto rilevato — pronto per l'analisi")
                if st.button("▶  AVVIA ANALISI", use_container_width=True):
                    st.session_state["phase"] = "analysis"
                    st.rerun()
            else:
                st.warning("⚠  Nessun volto rilevato — riprova con una luce migliore")


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif phase == "analysis":
    card("ANALISI IN CORSO")
    img_col, status_col = st.columns([1, 1])

    raw_pil    = st.session_state["raw_pil"]
    faces      = st.session_state["faces"]
    face_found = st.session_state["face_found"]

    with img_col:
        if st.session_state.get("gif") is None and face_found:
            st.session_state["gif"] = create_scanning_gif(raw_pil, faces)
        if face_found:
            st.image(st.session_state["gif"], use_container_width=True)
        else:
            st.image(st.session_state["annotated"], use_container_width=True)

    with status_col:
        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace; font-size:.8rem;
                    color:#00e5ff; letter-spacing:2px; margin-bottom:1.5rem;">
            ELABORAZIONE PARAMETRI BIOMETRICI...
        </div>
        """, unsafe_allow_html=True)

        progress_bar = st.progress(0, text="Inizializzazione…")
        status_text  = st.empty()

        steps_labels = [
            (15,  "Rilevamento geometria facciale…"),
            (35,  "Estrazione feature cromatiche…"),
            (55,  "Analisi morfologica strutturale…"),
            (75,  "Classificazione modello ML…"),
            (90,  "Calcolo appartenenza etnica…"),
            (100, "Completato."),
        ]

        # Run the actual classifier
        with st.spinner(""):
            results = classifier(raw_pil, top_k=10)

        # Animate the progress bar after classification
        for pct, label in steps_labels:
            progress_bar.progress(pct, text=label)
            status_text.markdown(
                f'<span style="font-family:\'Share Tech Mono\',monospace; '
                f'font-size:.7rem; color:#3a9a6a; letter-spacing:1px;">{label}</span>',
                unsafe_allow_html=True,
            )
            time.sleep(0.4)

        st.session_state["results"] = results
        st.session_state["phase"]   = "results"

    card_end()
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif phase == "results":
    results      = st.session_state["results"]
    annotated_pil = st.session_state["annotated"]

    card("RISULTATI ANALISI")

    img_col, bar_col = st.columns([1, 1])

    with img_col:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace; font-size:.7rem; color:#3a6a9a; letter-spacing:2px; margin-bottom:.5rem;">IMMAGINE ANALIZZATA</div>', unsafe_allow_html=True)
        st.image(annotated_pil, use_container_width=True)

    with bar_col:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace; font-size:.7rem; color:#3a6a9a; letter-spacing:2px; margin-bottom:.75rem;">ETNIA STIMATA</div>', unsafe_allow_html=True)
        for r in results:
            label   = r["label"].replace("_"," ").title()
            score   = r["score"] * 100
            matched = match_label(r["label"])
            color   = ETHNICITY_MAP[matched]["color"] if matched else "#445577"
            bar_w   = int(score * 1.8)
            st.markdown(
                f'<div class="result-row" style="display:flex;align-items:center;gap:8px;">'
                f'<span style="width:10px;height:10px;border-radius:50%;background:{color};'
                f'display:inline-block;flex-shrink:0;box-shadow:0 0 6px {color};"></span>'
                f'<span style="width:145px;font-size:13px;font-family:\'Rajdhani\',sans-serif;'
                f'font-weight:600;color:#c0cce8;">{label}</span>'
                f'<div style="background:#1a243a;border-radius:4px;height:8px;width:180px;border:1px solid #2a3a5a;">'
                f'<div style="background:{color};width:{bar_w}px;height:8px;border-radius:4px;'
                f'box-shadow:0 0 8px {color}66;"></div></div>'
                f'<span style="font-size:12px;color:#5a7aa0;font-family:\'Share Tech Mono\',monospace;">'
                f'{score:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    card_end()

    # ── World map ──────────────────────────────────────────────────────────────
    # card("DISTRIBUZIONE GEOGRAFICA ETNIA")
    fig = go.Figure()
    for r in results:
        matched = match_label(r["label"])
        if not matched:
            continue
        score = r["score"] * 100
        info  = ETHNICITY_MAP[matched]
        fig.add_trace(go.Choropleth(
            locations=info["countries"],
            z=[score] * len(info["countries"]),
            zmin=0, zmax=100,
            colorscale=[[0, hex_to_rgba(info["color"], 0.15)], [1, info["color"]]],
            showscale=False,
            name=matched,
            hovertemplate=f"<b>%{{location}}</b><br>{matched}: {score:.1f}%<extra></extra>",
            marker_line_color="#1a2a4a",
            marker_line_width=0.5,
        ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(
            bgcolor="#0a0e1a",
            showframe=False, showcoastlines=False,
            showland=True,  landcolor="#111a32",
            showocean=True, oceancolor="#0a0e1a",
            showcountries=True, countrycolor="#1e2e50", showlakes=False,
        ),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)
    card_end()

    # ── Reset button ───────────────────────────────────────────────────────────
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        if st.button("↩  NUOVA ANALISI", use_container_width=True):
            for k in ["phase","photo_id","results","annotated","gif","faces","raw_pil","face_found"]:
                st.session_state.pop(k, None)
            st.session_state["phase"] = "photo"
            st.rerun()