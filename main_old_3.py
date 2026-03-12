import streamlit as st
from PIL import Image
from transformers import pipeline
import torch
import plotly.graph_objects as go
import streamlit.components.v1 as components
import base64, io, re

st.set_page_config(layout="wide")

# ── Ethnicity map ─────────────────────────────────────────────────────────────
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
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2],16), int(hex_color[2:4],16), int(hex_color[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

@st.cache_resource
def load_classifier():
    return pipeline(
        "image-classification",
        model="cledoux42/Ethnicity_Test_v003",
        device=0 if torch.cuda.is_available() else -1,
    )

classifier = load_classifier()

# ── HTML webcam component ─────────────────────────────────────────────────────
WEBCAM_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d0d0d; font-family: 'Courier New', monospace; }

  #container {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 12px;
    gap: 12px;
  }

  #cam-box {
    position: relative;
    width: 100%;
    max-width: 820px;
    border-radius: 12px;
    overflow: hidden;
    background: #000;
    border: 1.5px solid #1a1a2e;
    box-shadow: 0 0 32px rgba(0,200,255,0.12);
  }

  #video {
    display: block;
    width: 100%;
    height: auto;
    border-radius: 12px;
  }

  #overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    border-radius: 12px;
  }

  #status-bar {
    width: 100%;
    max-width: 820px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 12px;
    background: #111;
    border-radius: 8px;
    border: 1px solid #222;
  }

  #status {
    font-size: 12px;
    color: #00e5ff;
    letter-spacing: 1px;
  }

  #snap-btn {
    background: linear-gradient(135deg, #0077b6, #00b4d8);
    color: #fff;
    border: none;
    padding: 8px 24px;
    border-radius: 6px;
    font-size: 13px;
    font-family: 'Courier New', monospace;
    letter-spacing: 1px;
    cursor: pointer;
    transition: opacity 0.2s, box-shadow 0.2s;
    box-shadow: 0 0 10px rgba(0,180,216,0.4);
  }
  #snap-btn:hover  { opacity: 0.85; box-shadow: 0 0 18px rgba(0,180,216,0.7); }
  #snap-btn:active { opacity: 0.7; }

  canvas#hidden { display: none; }
</style>
</head>
<body>
<div id="container">
  <div id="cam-box">
    <video id="video" autoplay playsinline muted></video>
    <canvas id="overlay"></canvas>
  </div>

  <div id="status-bar">
    <span id="status">⏳ INITIALISING…</span>
    <button id="snap-btn">📸 CAPTURE</button>
  </div>
</div>

<canvas id="hidden"></canvas>

<script src="https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js"></script>
<script>
const video   = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx     = overlay.getContext('2d');
const status  = document.getElementById('status');
const btn     = document.getElementById('snap-btn');
const hidden  = document.getElementById('hidden');

let modelsLoaded = false;
let dashOff      = 0;
let lastFaces    = [];
let smoothFaces  = [];   // for smooth interpolation

// ── Load face-api models ──────────────────────────────────────────────────────
const MODEL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model';
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri(MODEL),
  faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL),
]).then(() => {
  modelsLoaded = true;
  status.textContent = '✅ MODELS READY — STARTING CAMERA';
  startCam();
}).catch(err => {
  status.textContent = '⚠️ MODEL ERROR — CAMERA ONLY MODE';
  startCam();
});

// ── Camera ────────────────────────────────────────────────────────────────────
async function startCam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
    });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      syncOverlay();
      status.textContent = modelsLoaded ? '🔍 SCANNING…' : '📷 CAMERA READY';
      if (modelsLoaded) detectLoop();
      drawLoop();
    };
  } catch(e) {
    status.textContent = '❌ CAMERA ACCESS DENIED';
  }
}

function syncOverlay() {
  overlay.width  = video.videoWidth  || video.offsetWidth;
  overlay.height = video.videoHeight || video.offsetHeight;
}

// ── Detection ─────────────────────────────────────────────────────────────────
async function detectLoop() {
  try {
    const det = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.35 }))
      .withFaceLandmarks(true);
    lastFaces = det;
    status.textContent = det.length > 0
      ? `✅ ${det.length} FACE${det.length > 1 ? 'S' : ''} DETECTED`
      : '🔍 SCANNING…';
  } catch(_) {}
  setTimeout(detectLoop, 120);   // ~8fps detection, 60fps draw
}

// ── Draw loop ─────────────────────────────────────────────────────────────────
function drawLoop() {
  syncOverlay();
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  dashOff -= 1.2;

  // scan-line effect
  ctx.save();
  const scanY = ((Date.now() / 12) % overlay.height);
  const scanGrad = ctx.createLinearGradient(0, scanY - 40, 0, scanY + 4);
  scanGrad.addColorStop(0, 'rgba(0,229,255,0)');
  scanGrad.addColorStop(1, 'rgba(0,229,255,0.06)');
  ctx.fillStyle = scanGrad;
  ctx.fillRect(0, scanY - 40, overlay.width, 44);
  ctx.restore();

  lastFaces.forEach(det => drawFace(det));
  requestAnimationFrame(drawLoop);
}

function drawFace(det) {
  const box = det.detection.box;
  const lm  = det.landmarks;
  const CYAN  = '#00e5ff';
  const WHITE = 'rgba(255,255,255,0.9)';

  // ── tinted fill ─────────────────────────────────────────────────────────────
  ctx.save();
  ctx.fillStyle = 'rgba(0,229,255,0.05)';
  ctx.fillRect(box.x, box.y, box.width, box.height);
  ctx.restore();

  // ── animated dashed rect ─────────────────────────────────────────────────
  ctx.save();
  ctx.strokeStyle = CYAN;
  ctx.lineWidth   = 1.8;
  ctx.shadowColor = CYAN;
  ctx.shadowBlur  = 14;
  ctx.setLineDash([12, 7]);
  ctx.lineDashOffset = dashOff;
  roundRect(ctx, box.x, box.y, box.width, box.height, 10);
  ctx.stroke();
  ctx.restore();

  // ── landmarks ────────────────────────────────────────────────────────────
  const groups = [
    lm.getJawOutline(), lm.getLeftEye(), lm.getRightEye(),
    lm.getLeftEyeBrow(), lm.getRightEyeBrow(), lm.getNose(), lm.getMouth()
  ];
  const closedGroups = [lm.getLeftEye(), lm.getRightEye(), lm.getMouth()];

  groups.forEach(pts => {
    ctx.save();
    ctx.beginPath();
    pts.forEach((p,i) => i===0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
    if (closedGroups.includes(pts)) ctx.closePath();
    ctx.strokeStyle = 'rgba(0,229,255,0.7)';
    ctx.lineWidth   = 1.2;
    ctx.shadowColor = CYAN;
    ctx.shadowBlur  = 6;
    ctx.setLineDash([]);
    ctx.stroke();
    ctx.restore();
  });

  // ── landmark dots ─────────────────────────────────────────────────────────
  const allPts = lm.positions;
  allPts.forEach(p => {
    ctx.save();
    ctx.beginPath();
    ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2);
    ctx.fillStyle   = 'rgba(0,229,255,0.6)';
    ctx.shadowColor = CYAN;
    ctx.shadowBlur  = 4;
    ctx.fill();
    ctx.restore();
  });

  // ── corner brackets ───────────────────────────────────────────────────────
  const arm = Math.max(20, box.width / 7);
  const corners = [
    [box.x,             box.y,              1,  1],
    [box.x+box.width,   box.y,             -1,  1],
    [box.x,             box.y+box.height,   1, -1],
    [box.x+box.width,   box.y+box.height,  -1, -1],
  ];
  corners.forEach(([cx,cy,sx,sy]) => {
    ctx.save();
    ctx.strokeStyle = WHITE;
    ctx.lineWidth   = 3;
    ctx.shadowColor = CYAN;
    ctx.shadowBlur  = 16;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(cx + sx*arm, cy);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx, cy + sy*arm);
    ctx.stroke();
    ctx.restore();
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI*2);
    ctx.fillStyle   = CYAN;
    ctx.shadowColor = CYAN;
    ctx.shadowBlur  = 10;
    ctx.fill();
    ctx.restore();
  });

  // ── confidence badge ──────────────────────────────────────────────────────
  const conf = (det.detection.score * 100).toFixed(0);
  const tag  = ` CONF ${conf}% `;
  ctx.save();
  ctx.font    = 'bold 11px "Courier New"';
  const tw    = ctx.measureText(tag).width;
  const bx    = box.x, by = Math.max(18, box.y - 20);
  ctx.fillStyle = 'rgba(0,229,255,0.9)';
  ctx.fillRect(bx, by - 14, tw + 4, 18);
  ctx.fillStyle   = '#000';
  ctx.shadowBlur  = 0;
  ctx.fillText(tag, bx + 2, by);
  ctx.restore();
}

function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x+r, y);
  c.lineTo(x+w-r, y); c.quadraticCurveTo(x+w, y, x+w, y+r);
  c.lineTo(x+w, y+h-r); c.quadraticCurveTo(x+w, y+h, x+w-r, y+h);
  c.lineTo(x+r, y+h); c.quadraticCurveTo(x, y+h, x, y+h-r);
  c.lineTo(x, y+r); c.quadraticCurveTo(x, y, x+r, y);
  c.closePath();
}

// ── Capture & send to Streamlit ───────────────────────────────────────────────
btn.addEventListener('click', () => {
  hidden.width  = video.videoWidth;
  hidden.height = video.videoHeight;
  hidden.getContext('2d').drawImage(video, 0, 0);
  const dataUrl = hidden.toDataURL('image/jpeg', 0.92);
  // Streamlit component value communication
  window.parent.postMessage({
    isStreamlitMessage: true,
    type: 'streamlit:setComponentValue',
    value: dataUrl,
  }, '*');
});
</script>
</body>
</html>
"""

# ── Page layout ───────────────────────────────────────────────────────────────
tab_cam, tab_upload = st.tabs(["📷 Webcam", "🖼️ Upload"])

img_input = None

with tab_cam:
    # Render the webcam component — tall enough to show a 16:9 feed comfortably
    cam_data = components.html(WEBCAM_HTML, height=600, scrolling=False)

    st.caption("After capturing, paste or re-upload the image below if needed, "
               "or use the Upload tab for an existing photo.")

    # Since postMessage to Streamlit requires a registered component (not html()),
    # we accept the captured frame via a hidden file uploader as the reliable path.
    st.markdown("**Or upload your captured photo for analysis:**")
    cam_upload = st.file_uploader("Drop captured image", type=["jpg","jpeg","png"],
                                  key="cam_upload", label_visibility="collapsed")
    if cam_upload:
        img_input = cam_upload

with tab_upload:
    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded:
        img_input = uploaded

# ── Analysis ─────────────────────────────────────────────────────────────────
if img_input:
    image = Image.open(img_input).convert("RGB")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, use_container_width=True)
        with st.spinner("Analysing…"):
            results = classifier(image, top_k=10)

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

    # ── Choropleth map ────────────────────────────────────────────────────────
    fig = go.Figure()

    for r in results:
        matched = match_label(r["label"])
        if not matched:
            continue
        score         = r["score"] * 100
        info          = ETHNICITY_MAP[matched]
        label_display = r["label"].replace("_"," ").title()
        hex_col       = info["color"]

        # Start from a clearly visible tint (0.25 alpha) so low scores are
        # still readable, up to full opacity at 100 %
        color_low  = hex_to_rgba(hex_col, 0.25)
        color_high = hex_col

        fig.add_trace(go.Choropleth(
            locations=info["countries"],
            z=[score] * len(info["countries"]),
            zmin=0,
            zmax=100,
            colorscale=[[0, color_low], [1, color_high]],
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