import streamlit as st
from PIL import Image
from transformers import pipeline
import torch
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# Each ethnicity: countries + a distinct color
ETHNICITY_MAP = {
    "European": {
        "color": "#4A90D9",
        "countries": ["DEU", "FRA", "GBR", "ITA", "ESP", "POL", "ROU", "NLD", "BEL", "SWE",
                      "CZE", "GRC", "PRT", "HUN", "AUT", "CHE", "BGR", "DNK", "FIN", "SVK",
                      "NOR", "IRL", "HRV", "BIH", "ALB", "LTU", "SVN", "LVA", "EST", "MKD",
                      "LUX", "MNE", "MLT", "ISL", "RUS", "UKR", "BLR", "MDA", "SRB", "USA",
                      "CAN", "AUS", "NZL"],
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

img_input = st.camera_input("") or st.file_uploader("", type=["jpg", "jpeg", "png"])

if img_input:
    image = Image.open(img_input).convert("RGB")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, width=280)
        with st.spinner("Analysing..."):
            results = classifier(image, top_k=10)
        for r in results:
            label = r["label"].replace("_", " ").title()
            score = r["score"] * 100
            matched = match_label(r["label"])
            color = ETHNICITY_MAP[matched]["color"] if matched else "#888888"
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
        score = r["score"] * 100
        info = ETHNICITY_MAP[matched]
        countries = info["countries"]
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