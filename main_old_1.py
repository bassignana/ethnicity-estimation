import streamlit as st
from PIL import Image
from transformers import pipeline
import torch
import plotly.graph_objects as go
import streamlit.components.v1 as components
import base64
import io

st.set_page_config(layout="wide")

# Each ethnicity: countries + a distinct color
ETHNICITY_MAP = {
    "European": {
        "color": "#4A90D9",
        # Only actual European countries (removed USA, CAN, AUS, NZL)
        "countries": [
            "DEU", "FRA", "GBR", "ITA", "ESP", "POL", "ROU", "NLD", "BEL", "SWE",
            "CZE", "GRC", "PRT", "HUN", "AUT", "CHE", "BGR", "DNK", "FIN", "SVK",
            "NOR", "IRL", "HRV", "BIH", "ALB", "LTU", "SVN", "LVA", "EST", "MKD",
            "LUX", "MNE", "MLT", "ISL", "RUS", "UKR", "BLR", "MDA", "SRB",
            "AND", "LIE", "SMR", "VAT", "MCO", "GEO", "ARM", "AZE",
        ],
    },
    "Latino Hispanic": {
        "color": "#E8A838",
        "countries": ["MEX", "COL", "ARG", "PER", "VEN", "CHL", "ECU", "GTM", "CUB", "BOL",
                      "DOM", "HND", "PRY", "SLV", "NIC", "CRI", "PAN", "URY", "BRA"],
    },
    "South Asian": {
        "color": "#7ED321",
        "countries": ["IND", "PAK", "BGD", "LKA", "NPL", "BTN", "MDV"],
    },
    "East Asian": {
        "color": "#F5A0C0",
        "countries": ["CHN", "JPN", "KOR", "MNG", "TWN", "HKG", "MAC", "VNM", "THA", "IDN",
                      "PHL", "MYS", "MMR", "KHM", "LAO", "SGP", "BRN", "TLS"],
    },
    "African": {
        "color": "#C0703A",
        "countries": ["NGA", "ETH", "COD", "TZA", "KEN", "ZAF", "UGA", "GHA", "MOZ", "AGO",
                      "MDG", "CMR", "CIV", "SEN", "ZMB", "ZWE", "MLI", "BFA", "NER", "SOM",
                      "EGY", "DZA", "MAR", "LBY", "TUN", "SDN"],
    },
    "Middle Eastern": {
        "color": "#9B59B6",
        "countries": ["SAU", "IRN", "IRQ", "SYR", "JOR", "LBN", "ISR", "YEM", "OMN", "ARE",
                      "KWT", "QAT", "BHR", "TUR"],
    },
}

LABEL_MAPPING = {
    "caucasian": "European",
    "white": "European",
    "european": "European",
    "hispanic": "Latino Hispanic",
    "latino": "Latino Hispanic",
    "latino hispanic": "Latino Hispanic",
    "indian": "South Asian",
    "south asian": "South Asian",
    "asian": "East Asian",
    "east asian": "East Asian",
    "southeast asian": "East Asian",
    "black": "African",
    "african": "African",
    "african american": "African",
    "middle eastern": "Middle Eastern",
    "arab": "Middle Eastern",
}

def match_label(raw_label: str):
    raw = raw_label.lower().replace("_", " ")
    for key, val in LABEL_MAPPING.items():
        if key in raw:
            return val
    return None

@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="cledoux42/Ethnicity_Test_v003",
        device=0 if torch.cuda.is_available() else -1
    )

classifier = load_model()

# ── Webcam component with face-detection animation ──────────────────────────
WEBCAM_HTML = """
<style>
  #webcam-wrapper {
    position: relative;
    display: inline-block;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.18);
  }
  #video { display: block; width: 320px; border-radius: 12px; }
  #overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
  }
  #controls { margin-top: 10px; text-align: center; }
  #snap-btn {
    background: #4A90D9;
    color: white;
    border: none;
    padding: 8px 28px;
    border-radius: 8px;
    font-size: 15px;
    cursor: pointer;
    transition: background 0.2s;
  }
  #snap-btn:hover { background: #2e6fad; }
  #status {
    font-size: 12px;
    color: #666;
    margin-top: 6px;
    min-height: 18px;
    text-align: center;
  }
</style>

<div id="webcam-wrapper">
  <video id="video" autoplay playsinline muted></video>
  <canvas id="overlay"></canvas>
</div>
<div id="controls">
  <button id="snap-btn">📸 Capture</button>
  <div id="status">Loading face detection…</div>
</div>
<canvas id="hidden-canvas" style="display:none"></canvas>

<!-- face-api.js from CDN -->
<script src="https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js"></script>

<script>
const video    = document.getElementById('video');
const overlay  = document.getElementById('overlay');
const ctx      = overlay.getContext('2d');
const status   = document.getElementById('status');
const snapBtn  = document.getElementById('snap-btn');
const hidden   = document.getElementById('hidden-canvas');

let modelsReady = false;
let animFrame;
let dashOffset = 0;

// ── load models ─────────────────────────────────────────────────────────────
const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model';

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
  faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL),
]).then(() => {
  modelsReady = true;
  status.textContent = 'Face detection ready ✓';
  startCamera();
}).catch(e => {
  status.textContent = 'Model load error – detection disabled';
  startCamera();          // still allow capture without overlay
  console.warn(e);
});

// ── camera ───────────────────────────────────────────────────────────────────
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      overlay.width  = video.videoWidth  || video.offsetWidth;
      overlay.height = video.videoHeight || video.offsetHeight;
      if (modelsReady) detectLoop();
    };
  } catch(e) {
    status.textContent = 'Camera access denied';
  }
}

// ── detection loop ───────────────────────────────────────────────────────────
async function detectLoop() {
  const detection = await faceapi
    .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.4 }))
    .withFaceLandmarks(true);

  ctx.clearRect(0, 0, overlay.width, overlay.height);
  dashOffset -= 2;   // animate the dashes

  if (detection) {
    drawFaceContour(detection);
    status.textContent = '✅ Face detected';
  } else {
    status.textContent = 'No face detected';
  }

  animFrame = requestAnimationFrame(detectLoop);
}

// ── drawing ──────────────────────────────────────────────────────────────────
function drawFaceContour(detection) {
  const lm   = detection.landmarks;
  const box  = detection.detection.box;
  const pts  = lm.getJawOutline();          // 17 jaw points
  const nose = lm.getNose();
  const lEye = lm.getLeftEye();
  const rEye = lm.getRightEye();
  const lBrow= lm.getLeftEyeBrow();
  const rBrow= lm.getRightEyeBrow();
  const mouth= lm.getMouth();

  const neon = '#00e5ff';
  const glow  = 'rgba(0,229,255,0.25)';

  // ── glow bounding box ────────────────────────────────────────────────────
  ctx.save();
  ctx.strokeStyle = neon;
  ctx.lineWidth   = 2;
  ctx.shadowColor = neon;
  ctx.shadowBlur  = 18;
  ctx.setLineDash([10, 6]);
  ctx.lineDashOffset = dashOffset;
  roundRect(ctx, box.x, box.y, box.width, box.height, 14);
  ctx.stroke();
  ctx.restore();

  // ── landmark paths ────────────────────────────────────────────────────────
  const groups = [pts, lEye, rEye, lBrow, rBrow, mouth];
  groups.forEach(group => {
    ctx.save();
    ctx.beginPath();
    group.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
    if (group === lEye || group === rEye || group === mouth) ctx.closePath();
    ctx.strokeStyle = neon;
    ctx.lineWidth   = 1.5;
    ctx.shadowColor = neon;
    ctx.shadowBlur  = 10;
    ctx.setLineDash([]);
    ctx.stroke();
    ctx.restore();
  });

  // ── nose bridge ───────────────────────────────────────────────────────────
  ctx.save();
  ctx.beginPath();
  nose.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
  ctx.strokeStyle = neon;
  ctx.lineWidth   = 1.5;
  ctx.shadowColor = neon;
  ctx.shadowBlur  = 10;
  ctx.setLineDash([]);
  ctx.stroke();
  ctx.restore();

  // ── corner accents ────────────────────────────────────────────────────────
  const corners = [
    [box.x,             box.y,              1,  1],
    [box.x + box.width, box.y,             -1,  1],
    [box.x,             box.y + box.height,  1, -1],
    [box.x + box.width, box.y + box.height, -1, -1],
  ];
  const L = 18;
  corners.forEach(([cx, cy, dx, dy]) => {
    ctx.save();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth   = 3;
    ctx.shadowColor = neon;
    ctx.shadowBlur  = 14;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(cx + dx * L, cy);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx, cy + dy * L);
    ctx.stroke();
    ctx.restore();
  });

  // ── confidence score ─────────────────────────────────────────────────────
  const conf = (detection.detection.score * 100).toFixed(0);
  ctx.save();
  ctx.fillStyle   = neon;
  ctx.font        = 'bold 11px monospace';
  ctx.shadowColor = neon;
  ctx.shadowBlur  = 8;
  ctx.fillText(`CONF ${conf}%`, box.x + 4, box.y - 6);
  ctx.restore();
}

// ── helpers ───────────────────────────────────────────────────────────────
function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y);
  c.lineTo(x + w - r, y);
  c.quadraticCurveTo(x + w, y, x + w, y + r);
  c.lineTo(x + w, y + h - r);
  c.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  c.lineTo(x + r, y + h);
  c.quadraticCurveTo(x, y + h, x, y + h - r);
  c.lineTo(x, y + r);
  c.quadraticCurveTo(x, y, x + r, y);
  c.closePath();
}

// ── capture ──────────────────────────────────────────────────────────────────
snapBtn.addEventListener('click', () => {
  hidden.width  = video.videoWidth;
  hidden.height = video.videoHeight;
  hidden.getContext('2d').drawImage(video, 0, 0);
  const dataUrl = hidden.toDataURL('image/jpeg', 0.92);
  // Send to Streamlit via query-param trick
  window.parent.postMessage({ type: 'streamlit:setComponentValue', value: dataUrl }, '*');
});
</script>
"""

# ── Streamlit layout ─────────────────────────────────────────────────────────
tab_cam, tab_upload = st.tabs(["📷 Webcam", "🖼️ Upload"])

img_input = None

with tab_cam:
    cam_result = components.html(WEBCAM_HTML, height=420, scrolling=False)
    # Fallback: also allow st.camera_input if JS component doesn't post back
    st.markdown("---")
    st.caption("Or use the built-in camera capture below (no face-detection overlay):")
    cam_fallback = st.camera_input("", label_visibility="collapsed")
    if cam_fallback:
        img_input = cam_fallback

with tab_upload:
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded:
        img_input = uploaded

# ── Analysis ─────────────────────────────────────────────────────────────────
if img_input:
    image = Image.open(img_input).convert("RGB")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, width=280)
        with st.spinner("Analysing…"):
            results = classifier(image, top_k=10)
        for r in results:
            label   = r["label"].replace("_", " ").title()
            score   = r["score"] * 100
            matched = match_label(r["label"])
            color   = ETHNICITY_MAP[matched]["color"] if matched else "#888888"
            st.markdown(
                f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
                f'background:{color};margin-right:8px;"></span>**{label}**: {score:.1f}%',
                unsafe_allow_html=True,
            )

    fig = go.Figure()

    for r in results:
        matched = match_label(r["label"])
        if not matched:
            continue
        score        = r["score"] * 100
        info         = ETHNICITY_MAP[matched]
        countries    = info["countries"]
        label_display = r["label"].replace("_", " ").title()

        fig.add_trace(go.Choropleth(
            locations=countries,
            z=[score] * len(countries),
            zmin=0,
            zmax=100,
            colorscale=[[0, "rgba(255,255,255,0)"], [1, info["color"]]],
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
            bgcolor="white",
            showframe=False,
            showcoastlines=False,
            showland=True,
            landcolor="#eeeeee",
            showocean=True,
            oceancolor="#ddeeff",
            showcountries=True,
            countrycolor="#cccccc",
            showlakes=False,
        ),
        height=440,
    )

    with col2:
        st.plotly_chart(fig, use_container_width=True)