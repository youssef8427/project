# ============================================================
# AUTOPOWER AI — FINAL FIX
# ONE CELL
#
# This cell:
# 1. Installs the required Streamlit packages
# 2. Exports the trained model/artifacts if they already exist
# 3. Creates a robust Streamlit app
# 4. Fixes Manufacturer / Brand KeyError
# 5. Supports 1950–2026 model years
# 6. Uses easy searchable vehicle selection
# 7. Uses a collapsed Streamlit sidebar
# 8. Generates a procedural 3D-style vehicle preview
# 9. Keeps the trained ML pipeline unchanged
# ============================================================

!pip -q install streamlit pandas numpy scikit-learn joblib plotly category_encoders xgboost

from pathlib import Path
import os
import textwrap

# ============================================================
# 1. CREATE MODEL DIRECTORY
# ============================================================

MODEL_DIR = Path("/content/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Checking trained model objects...")

# ============================================================
# 2. EXPORT EXISTING NOTEBOOK MODEL OBJECTS
# ============================================================

# The original notebook already creates these variables.
# If they exist, save them automatically.

exported = []

if "best_model" in globals():
    import joblib
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    exported.append("best_model.joblib")

if "df" in globals():
    import joblib
    joblib.dump(df, MODEL_DIR / "dataset.joblib")
    exported.append("dataset.joblib")

elif "dataset" in globals():
    import joblib
    joblib.dump(dataset, MODEL_DIR / "dataset.joblib")
    exported.append("dataset.joblib")

if "results_df" in globals():
    import joblib
    joblib.dump(results_df, MODEL_DIR / "results_df.joblib")
    exported.append("results_df.joblib")

if "target_col" in globals():
    import joblib
    joblib.dump(target_col, MODEL_DIR / "target_col.joblib")
    exported.append("target_col.joblib")

elif "TARGET_COL" in globals():
    import joblib
    joblib.dump(TARGET_COL, MODEL_DIR / "target_col.joblib")
    exported.append("target_col.joblib")

if "feature_importance" in globals():
    import joblib
    joblib.dump(feature_importance, MODEL_DIR / "feature_importance.joblib")
    exported.append("feature_importance.joblib")

if "trained_models" in globals():
    import joblib
    joblib.dump(trained_models, MODEL_DIR / "trained_models.joblib")
    exported.append("trained_models.joblib")

print()
print("Exported from notebook:")
for x in exported:
    print("  ✓", x)

# ============================================================
# 3. CHECK REQUIRED ARTIFACTS
# ============================================================

required = [
    "best_model.joblib",
    "dataset.joblib",
]

missing = [
    x for x in required
    if not (MODEL_DIR / x).exists()
]

if missing:
    print()
    print("❌ The trained model is not currently available in this notebook.")
    print()
    print("Missing:")
    for x in missing:
        print("  ✗", x)
    print()
    print("The Streamlit app cannot create the trained model from nothing.")
    print("Run your existing training cells first, then run this cell again.")
    raise RuntimeError(
        "Trained model objects were not found. "
        "Run the model-training/export cells first."
    )

print()
print("✓ Model artifacts are ready.")

# ============================================================
# 4. FULL STREAMLIT APPLICATION
# ============================================================

STREAMLIT_APP_CODE = r'''
"""
AUTOPOWER AI
Machine Learning Automotive Horsepower Predictor

Final robust version.

The trained ML model is loaded exactly as exported by the notebook.
The application does not retrain or modify the trained pipeline.
"""

import hashlib
import math
import textwrap
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# MARKDOWN / HTML RENDER FIX
#
# Streamlit's markdown renderer follows CommonMark: any line
# indented 4+ spaces is treated as a literal (fenced) code block,
# not HTML — even with unsafe_allow_html=True. This file builds
# HTML inside heavily-indented f-strings WITH blank lines between
# nested <div> tags. Each blank line resets the block, so even
# after removing the *common* indentation (textwrap.dedent), the
# inner divs still carried enough leftover indent to retrigger the
# bug on the next chunk — which is why some cards still rendered
# as a raw, copy-able code box instead of styled HTML.
#
# Real fix: strip leading/trailing whitespace from EVERY line
# individually before handing it to Streamlit. HTML doesn't care
# about whitespace between tags, so this is always safe here.
# This is a drop-in patch — no call sites need to change.
# ============================================================

_original_st_markdown = st.markdown


def _safe_markdown(body, *args, **kwargs):

    if isinstance(body, str) and kwargs.get("unsafe_allow_html"):
        body = "\n".join(
            line.strip()
            for line in body.strip("\n").split("\n")
        )

    return _original_st_markdown(body, *args, **kwargs)


st.markdown = _safe_markdown


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "AUTOPOWER AI"
APP_TAGLINE = "Intelligent Automotive Performance"

try:
    APP_DIR = Path(__file__).resolve().parent
except Exception:
    APP_DIR = Path.cwd()

MODEL_DIRS = [
    APP_DIR / "models",
    Path("/content/models"),
    Path("models"),
]

TARGET_COL_DEFAULT = "Power (hp)"

# Exact model input contract
HIGH_CARDINALITY_COLS = [
    "Brand_Manufacturer"
]

LOW_CARDINALITY_COLS = [
    "Origin Country",
    "Body Type",
    "Additional Type",
    "gear_type",
]

NUMERIC_COLS = [
    "Approx Cost",
    "Model Year",
    "Weight",
    "Fuel Econ (L/100km)",
    "Fuel Econ (km/L)",
    "Performance 0-100 kph (sec)",
    "Top speed (kph)",
    "gear_count",
]

REQUIRED_MODEL_COLUMNS = (
    HIGH_CARDINALITY_COLS
    + LOW_CARDINALITY_COLS
    + NUMERIC_COLS
)

PERFORMANCE_TIERS = [
    (0, "City Cruiser", "🚗"),
    (120, "Daily Driver", "🛣️"),
    (200, "Sporty", "🔥"),
    (300, "Performance", "🏎️"),
    (450, "Supercar Territory", "🚀"),
    (600, "Hypercar Beast", "👑"),
]

GEAR_TYPE_LABELS = {
    "A": "Automatic",
    "M": "Manual",
    "AM": "Automated Manual",
    "AT": "Automatic",
    "CVT": "CVT",
}

BODY_TYPE_ICONS = {
    "SUV": "🚙",
    "Sedan": "🚗",
    "Coupe": "🏎️",
    "Hatchback": "🚗",
    "Convertible": "🏎️",
    "Pickup": "🛻",
    "Van": "🚐",
    "Wagon": "🚙",
    "Crossover": "🚙",
    "Minivan": "🚐",
}

COMPARISON_METRICS = [
    ("Power (hp)", "higher", "HP"),
    ("Weight", "lower_neutral", "kg"),
    ("Top speed (kph)", "higher", "km/h"),
    ("Performance 0-100 kph (sec)", "lower", "s"),
    ("Fuel Econ (km/L)", "higher", "km/L"),
]


# ============================================================
# COLORS
# ============================================================

COLORS = {
    "bg": "#0A0C0F",
    "bg_alt": "#111418",
    "surface": "rgba(255,255,255,0.045)",
    "surface_border": "rgba(255,255,255,0.09)",
    "text": "#F2F4F6",
    "text_dim": "#8B939C",
    "accent": "#FF5A1F",
    "accent_soft": "rgba(255,90,31,0.15)",
    "telemetry": "#2FD4C0",
    "telemetry_soft": "rgba(47,212,192,0.15)",
    "redline": "#FF2D55",
    "divider": "rgba(255,255,255,0.08)",
}

CHART_SEQUENCE = [
    COLORS["accent"],
    COLORS["telemetry"],
    "#8C7BFF",
    "#FFC24B",
    "#4B9FFF",
    "#FF7BAC",
]


# ============================================================
# CSS
# ============================================================

def inject_css():

    st.markdown(
        f"""
        <style>

        @import url(
        'https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap'
        );

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(
                    ellipse 1200px 600px at 20% -10%,
                    rgba(255,90,31,0.08),
                    transparent 60%
                ),
                radial-gradient(
                    ellipse 900px 500px at 90% 0%,
                    rgba(47,212,192,0.06),
                    transparent 60%
                ),
                {COLORS["bg"]};
            color: {COLORS["text"]};
        }}

        #MainMenu, footer, header {{
            visibility: hidden;
        }}

        .block-container {{
            padding-top: 1.4rem;
            max-width: 1240px;
        }}

        .ap-display {{
            font-family: 'Bebas Neue', sans-serif;
            letter-spacing: 0.04em;
            line-height: 1;
        }}

        .ap-mono {{
            font-family: 'JetBrains Mono', monospace;
        }}

        .ap-eyebrow {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: {COLORS["telemetry"]};
        }}

        .ap-hero {{
            padding: 2.2rem;
            border-radius: 22px;
            background:
                linear-gradient(
                    180deg,
                    rgba(255,255,255,0.05),
                    rgba(255,255,255,0.015)
                );
            border: 1px solid {COLORS["surface_border"]};
            margin-bottom: 1.2rem;
            position: relative;
            overflow: hidden;
        }}

        .ap-hero h1 {{
            font-size: 3.1rem;
            margin: 0;
            color: {COLORS["text"]};
        }}

        .ap-hero h1 span {{
            color: {COLORS["accent"]};
        }}

        .ap-hero p.tagline {{
            font-size: 1rem;
            color: {COLORS["text_dim"]};
            margin: 0.3rem 0 0.7rem 0;
        }}

        .ap-hero p.sub {{
            max-width: 700px;
            color: {COLORS["text_dim"]};
            font-size: 0.9rem;
            line-height: 1.5;
        }}

        .ap-card {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["surface_border"]};
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 0.9rem;
        }}

        .ap-card-title {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            color: {COLORS["telemetry"]};
            margin-bottom: 0.75rem;
        }}

        .ap-result {{
            text-align: center;
            padding: 1.1rem 0 0.5rem 0;
        }}

        .ap-result .label {{
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.2em;
            font-size: 0.75rem;
            color: {COLORS["text_dim"]};
            text-transform: uppercase;
        }}

        .ap-result .hp {{
            font-family: 'Bebas Neue', sans-serif;
            font-size: 5.2rem;
            color: {COLORS["accent"]};
            line-height: 1;
            margin: 0.1rem 0;
        }}

        .ap-result .kw {{
            font-family: 'JetBrains Mono', monospace;
            color: {COLORS["telemetry"]};
            font-size: 0.92rem;
        }}

        .ap-tier {{
            display: inline-block;
            margin-top: 0.6rem;
            padding: 0.35rem 1rem;
            border-radius: 999px;
            background: {COLORS["accent_soft"]};
            border: 1px solid rgba(255,90,31,0.35);
            font-size: 0.88rem;
        }}

        .ap-metric {{
            text-align: center;
            padding: 0.7rem 0.4rem;
        }}

        .ap-metric .v {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.35rem;
            color: {COLORS["text"]};
        }}

        .ap-metric .l {{
            font-size: 0.7rem;
            color: {COLORS["text_dim"]};
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }}

        .ap-spec-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.42rem 0;
            border-bottom: 1px solid {COLORS["divider"]};
            font-size: 0.86rem;
        }}

        .ap-spec-row:last-child {{
            border-bottom: none;
        }}

        .ap-spec-row .k {{
            color: {COLORS["text_dim"]};
        }}

        .ap-spec-row .v {{
            font-family: 'JetBrains Mono', monospace;
            color: {COLORS["text"]};
        }}

        hr.ap-div {{
            border: none;
            border-top: 1px solid {COLORS["divider"]};
            margin: 0.9rem 0;
        }}

        .stButton > button {{
            background:
                linear-gradient(
                    135deg,
                    {COLORS["accent"]},
                    #E8460F
                );
            color: white;
            font-weight: 700;
            font-size: 1rem;
            letter-spacing: 0.03em;
            border: none;
            border-radius: 14px;
            padding: 0.8rem 1.1rem;
            width: 100%;
        }}

        .stButton > button:hover {{
            transform: translateY(-1px);
        }}

        /* Sidebar starts collapsed */
        section[data-testid="stSidebar"] {{
            background: {COLORS["bg_alt"]};
            border-right: 1px solid {COLORS["divider"]};
        }}

        /* Easier select boxes */
        div[data-baseweb="select"] > div {{
            border-radius: 12px !important;
            min-height: 46px !important;
        }}

        /* Number inputs */
        div[data-testid="stNumberInput"] input {{
            border-radius: 10px;
        }}

        .ap-footer {{
            text-align: center;
            color: {COLORS["text_dim"]};
            font-size: 0.78rem;
            padding: 1.6rem 0 1rem 0;
        }}

        .ap-disclaimer {{
            font-size: 0.78rem;
            color: {COLORS["text_dim"]};
            border-left: 3px solid {COLORS["accent"]};
            padding: 0.5rem 0.9rem;
            background: rgba(255,90,31,0.06);
            border-radius: 6px;
        }}

        .ap-winner-banner {{
            text-align: center;
            padding: 1.2rem;
            border-radius: 16px;
            background:
                linear-gradient(
                    135deg,
                    rgba(255,90,31,0.16),
                    rgba(47,212,192,0.10)
                );
            border: 1px solid rgba(255,90,31,0.35);
            margin: 0.6rem 0 1rem 0;
        }}

        .ap-winner-banner .t {{
            font-family: 'Bebas Neue', sans-serif;
            font-size: 2.1rem;
            color: {COLORS["accent"]};
        }}

        .ap-empty-state {{
            text-align: center;
            color: {COLORS["text_dim"]};
            padding: 2.4rem 1rem;
            border: 1px dashed {COLORS["surface_border"]};
            border-radius: 16px;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL DIRECTORY
# ============================================================

def find_model_dir():

    required = [
        "best_model.joblib",
        "dataset.joblib",
    ]

    for directory in MODEL_DIRS:

        if not directory.exists():
            continue

        if all((directory / file).exists() for file in required):
            return directory

    return None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(model_dir):

    return joblib.load(
        Path(model_dir) / "best_model.joblib"
    )


@st.cache_data(show_spinner=False)
def load_artifacts(model_dir):

    model_dir = Path(model_dir)

    dataset = joblib.load(
        model_dir / "dataset.joblib"
    )

    results_df = None

    if (model_dir / "results_df.joblib").exists():
        results_df = joblib.load(
            model_dir / "results_df.joblib"
        )

    target_col = TARGET_COL_DEFAULT

    if (model_dir / "target_col.joblib").exists():
        target_col = joblib.load(
            model_dir / "target_col.joblib"
        )

    feature_importance = None

    if (model_dir / "feature_importance.joblib").exists():
        feature_importance = joblib.load(
            model_dir / "feature_importance.joblib"
        )

    dataset = clean_dataset(dataset)

    return (
        dataset,
        results_df,
        target_col,
        feature_importance,
    )


# ============================================================
# DATASET NORMALIZATION
# ============================================================

def clean_dataset(dataset):

    df = dataset.copy()

    df = df.drop_duplicates().reset_index(drop=True)

    # --------------------------------------------------------
    # IMPORTANT FIX
    #
    # Some versions of the saved dataset have:
    #
    # Manufacturer
    # Brand
    #
    # Other versions only have:
    #
    # Brand_Manufacturer
    #
    # We support BOTH.
    # --------------------------------------------------------

    if "Manufacturer" not in df.columns:
        df["Manufacturer"] = ""

    if "Brand" not in df.columns:
        df["Brand"] = ""

    if "Brand_Manufacturer" in df.columns:

        combined = (
            df["Brand_Manufacturer"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        missing_manufacturer = (
            df["Manufacturer"]
            .fillna("")
            .astype(str)
            .str.strip()
            == ""
        )

        missing_brand = (
            df["Brand"]
            .fillna("")
            .astype(str)
            .str.strip()
            == ""
        )

        # Best-effort recovery:
        # split combined name at first space.
        split = combined.str.split(
            n=1,
            expand=True
        )

        if split.shape[1] >= 2:

            df.loc[
                missing_manufacturer,
                "Manufacturer"
            ] = split.loc[
                missing_manufacturer,
                0
            ]

            df.loc[
                missing_brand,
                "Brand"
            ] = split.loc[
                missing_brand,
                1
            ]

        elif split.shape[1] == 1:

            df.loc[
                missing_manufacturer,
                "Manufacturer"
            ] = split.loc[
                missing_manufacturer,
                0
            ]

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    for col in NUMERIC_COLS + ["Power (hp)"]:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if "Power (hp)" in df.columns:

        df = df.dropna(
            subset=["Power (hp)"]
        ).reset_index(drop=True)

    # --------------------------------------------------------
    # String columns
    # --------------------------------------------------------

    for col in [
        "Manufacturer",
        "Brand",
        "Brand_Manufacturer",
        "Origin Country",
        "Body Type",
        "Additional Type",
        "gear_type",
    ]:

        if col in df.columns:

            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    # --------------------------------------------------------
    # Rebuild Brand_Manufacturer
    # --------------------------------------------------------

    if (
        "Manufacturer" in df.columns
        and "Brand" in df.columns
    ):

        rebuilt = (
            df["Manufacturer"].fillna("").astype(str)
            + " "
            + df["Brand"].fillna("").astype(str)
        ).str.strip()

        if "Brand_Manufacturer" not in df.columns:

            df["Brand_Manufacturer"] = rebuilt

        else:

            empty_combined = (
                df["Brand_Manufacturer"]
                .fillna("")
                .astype(str)
                .str.strip()
                == ""
            )

            df.loc[
                empty_combined,
                "Brand_Manufacturer"
            ] = rebuilt.loc[empty_combined]

    # --------------------------------------------------------
    # Stable ID
    # --------------------------------------------------------

    df["_car_id"] = [
        f"car_{i}"
        for i in range(len(df))
    ]

    return df


# ============================================================
# HELPERS
# ============================================================

def safe_unique(
    df,
    column,
    fallback=None
):

    if column not in df.columns:
        return fallback or []

    values = []

    for value in df[column].dropna().unique():

        text = str(value).strip()

        if text:
            values.append(text)

    values = sorted(
        list(dict.fromkeys(values)),
        key=lambda x: x.lower()
    )

    return values if values else (fallback or [])


def safe_numeric_range(
    series,
    default=(0.0, 100.0)
):

    s = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if s.empty:
        return default

    lo = float(s.min())
    hi = float(s.max())

    if lo == hi:
        hi = lo + 1

    return lo, hi


def safe_value(value, fallback):

    try:

        if pd.isna(value):
            return fallback

        if str(value).strip() == "":
            return fallback

        return value

    except Exception:

        return fallback


def classify_performance(hp):

    result = PERFORMANCE_TIERS[0]

    for minimum, label, emoji in PERFORMANCE_TIERS:

        if hp >= minimum:
            result = (
                minimum,
                label,
                emoji
            )

    return result[1], result[2]


def brand_color(name):

    h = int(
        hashlib.md5(
            str(name).encode()
        ).hexdigest(),
        16
    )

    hue1 = h % 360
    hue2 = (hue1 + 46) % 360

    return (
        f"hsl({hue1},72%,58%)",
        f"hsl({hue2},72%,42%)",
    )


def normalize_series(
    values,
    higher_is_better=True
):

    values = np.array(
        values,
        dtype=float
    )

    lo = np.nanmin(values)
    hi = np.nanmax(values)

    if hi == lo:
        return np.full_like(
            values,
            50.0
        )

    result = (
        (values - lo)
        / (hi - lo)
        * 100
    )

    if not higher_is_better:
        result = 100 - result

    return result


def pearson_label(r):

    if r is None:
        return "Not available"

    try:

        if np.isnan(r):
            return "Not available"

    except Exception:
        pass

    a = abs(r)

    if a >= 0.9:
        strength = "Very strong"
    elif a >= 0.7:
        strength = "Strong"
    elif a >= 0.5:
        strength = "Moderate"
    elif a >= 0.3:
        strength = "Weak"
    else:
        strength = "Very weak / negligible"

    direction = (
        "positive"
        if r > 0
        else "negative"
        if r < 0
        else "no"
    )

    return f"{strength} {direction} correlation"


# ============================================================
# INPUT DATAFRAME
# ============================================================

def build_input_dataframe(values):

    if not values.get(
        "Brand_Manufacturer"
    ):

        manufacturer = str(
            values.get(
                "Manufacturer",
                ""
            )
        ).strip()

        brand = str(
            values.get(
                "Brand",
                ""
            )
        ).strip()

        values[
            "Brand_Manufacturer"
        ] = f"{manufacturer} {brand}".strip()

    row = {}

    for column in REQUIRED_MODEL_COLUMNS:

        row[column] = values.get(
            column,
            np.nan
        )

    return pd.DataFrame([row])


def predict_power(
    model,
    input_df
):

    prediction = model.predict(
        input_df
    )

    return float(
        np.asarray(
            prediction
        ).ravel()[0]
    )


# ============================================================
# HERO
# ============================================================

def render_hero():

    st.markdown(
        f"""
        <div class="ap-hero">

            <div class="ap-eyebrow">
                Machine Learning · Automotive Performance
            </div>

            <h1 class="ap-display">
                ⚡ AUTO<span>POWER</span> AI
            </h1>

            <p class="tagline">
                {APP_TAGLINE}
            </p>

            <p class="sub">
                Configure a vehicle and let the trained machine-learning
                model estimate its engine horsepower.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar(
    dataset,
    best_name
):

    with st.sidebar:

        st.markdown(
            """
            <div class="ap-eyebrow">
                AutoPower AI
            </div>

            <div class="ap-display"
                 style="font-size:1.5rem;">
                Dashboard
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "<hr class='ap-div'>",
            unsafe_allow_html=True
        )

        manufacturers = (
            dataset["Manufacturer"]
            .nunique()
            if "Manufacturer"
            in dataset.columns
            else 0
        )

        years = pd.to_numeric(
            dataset["Model Year"],
            errors="coerce"
        )

        if years.notna().any():

            min_year = int(
                years.min()
            )

            max_year = int(
                years.max()
            )

        else:

            min_year = 1950
            max_year = 2026

        st.markdown(
            f"""
            <div class="ap-metric">
                <div class="v">
                    {len(dataset):,}
                </div>
                <div class="l">
                    Cars
                </div>
            </div>

            <div class="ap-metric">
                <div class="v">
                    {manufacturers:,}
                </div>
                <div class="l">
                    Manufacturers
                </div>
            </div>

            <div class="ap-metric">
                <div class="v">
                    {min_year}–{max_year}
                </div>
                <div class="l">
                    Dataset Years
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "<hr class='ap-div'>",
            unsafe_allow_html=True
        )

        st.caption(
            f"Active model: **{best_name}**"
        )


# ============================================================
# 3D STYLE VEHICLE
# ============================================================

def box_mesh(
    x0, x1,
    y0, y1,
    z0, z1
):

    xs = [
        x0, x1, x1, x0,
        x0, x1, x1, x0
    ]

    ys = [
        y0, y0, y1, y1,
        y0, y0, y1, y1
    ]

    zs = [
        z0, z0, z0, z0,
        z1, z1, z1, z1
    ]

    i = [
        0,0,4,4,
        0,0,1,1,
        2,2,3,3
    ]

    j = [
        1,2,5,6,
        1,5,2,6,
        3,7,0,4
    ]

    k = [
        2,3,6,7,
        5,4,6,5,
        7,6,4,7
    ]

    return xs, ys, zs, i, j, k


def render_vehicle_visual(
    manufacturer,
    brand,
    body_type,
    weight,
    top_speed,
    accel,
    gear_type,
    height_px=300
):

    name = (
        f"{manufacturer} {brand}"
    ).strip()

    key = name.lower()

    body = (
        str(body_type)
        .lower()
    )

    color1, color2 = brand_color(
        name
    )

    sporty = any(
        word in key
        for word in [
            "amg",
            "m ",
            " m",
            "rs",
            "type r",
            "gti",
            "mustang",
            "corvette",
            "911",
            "supra",
            "gt",
            "r8",
            "huracan",
            "aventador"
        ]
    )

    luxury = any(
        word in key
        for word in [
            "s-class",
            "maybach",
            "7 series",
            "a8",
            "continental",
            "bentley",
            "rolls",
            "range rover"
        ]
    )

    offroad = any(
        word in key
        for word in [
            "wrangler",
            "land cruiser",
            "defender",
            "bronco",
            "patrol",
            "g-class",
            "g wagon"
        ]
    )

    length = 4.35
    width = 1.82
    base_h = 0.58
    ground = 0.15

    if (
        "suv" in body
        or "crossover" in body
    ):

        length = 4.55
        width = 1.92
        base_h = 0.88
        ground = 0.28

    elif "pickup" in body:

        length = 5.15
        width = 1.95
        base_h = 0.84
        ground = 0.30

    elif (
        "van" in body
        or "minivan" in body
    ):

        length = 4.85
        width = 1.95
        base_h = 1.10
        ground = 0.25

    elif "hatch" in body:

        length = 3.95
        width = 1.78
        base_h = 0.68
        ground = 0.14

    elif "wagon" in body:

        length = 4.65
        width = 1.83
        base_h = 0.72
        ground = 0.16

    elif (
        "coupe" in body
        or "convertible" in body
    ):

        length = 4.35
        width = 1.86
        base_h = 0.43
        ground = 0.12

    if sporty:

        ground = max(
            0.10,
            ground - 0.03
        )

        base_h *= 0.90
        length *= 1.03

    if luxury:

        length *= 1.06
        width *= 1.02

    if offroad:

        ground += 0.08
        width *= 1.03

    try:
        weight = float(weight)
    except Exception:
        weight = 1500.0

    bulk = np.clip(
        (weight - 1200)
        / 1800,
        -0.12,
        0.30
    )

    width *= (
        1 + bulk * 0.08
    )

    length *= (
        1 + bulk * 0.05
    )

    wheel_r = (
        0.39
        if (
            "suv" in body
            or "pickup" in body
            or offroad
        )
        else 0.32
    )

    fig = go.Figure()

    def add_box(
        x0, x1,
        y0, y1,
        z0, z1,
        color,
        opacity=1
    ):

        xs, ys, zs, ii, jj, kk = box_mesh(
            x0, x1,
            y0, y1,
            z0, z1
        )

        fig.add_trace(
            go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                i=ii,
                j=jj,
                k=kk,
                color=color,
                opacity=opacity,
                flatshading=True,
                lighting=dict(
                    ambient=0.45,
                    diffuse=0.75,
                    specular=0.85,
                    roughness=0.22
                ),
                showlegend=False,
                hoverinfo="skip"
            )
        )

    # Shadow
    theta = np.linspace(
        0,
        2 * np.pi,
        60
    )

    fig.add_trace(
        go.Scatter3d(
            x=(
                length / 2 + 0.25
            ) * np.cos(theta),
            y=(
                width / 2 + 0.18
            ) * np.sin(theta),
            z=np.zeros_like(theta),
            mode="lines",
            line=dict(
                color="rgba(0,0,0,0.35)",
                width=16
            ),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    # Main body
    add_box(
        -length / 2,
        length / 2,
        -width / 2,
        width / 2,
        ground,
        ground + base_h * 0.55,
        color1,
        0.98
    )

    # Lower sill
    add_box(
        -length * 0.42,
        length * 0.42,
        -width * 0.505,
        width * 0.505,
        ground + 0.02,
        ground + base_h * 0.18,
        color2,
        0.92
    )

    # Cabin
    if "convertible" not in body:

        roof_x0 = (
            -length * 0.23
            if "pickup" not in body
            else -length * 0.02
        )

        roof_x1 = length * 0.25

        if (
            "van" in body
            or "minivan" in body
        ):

            roof_x0 = -length * 0.32
            roof_x1 = length * 0.28

        roof_z0 = (
            ground
            + base_h * 0.55
        )

        roof_z1 = (
            roof_z0
            + base_h * 0.60
        )

        add_box(
            roof_x0,
            roof_x1,
            -width * 0.39,
            width * 0.39,
            roof_z0,
            roof_z1,
            "#151A20",
            0.95
        )

        add_box(
            roof_x0 + 0.05,
            roof_x1 - 0.05,
            -width * 0.32,
            width * 0.32,
            roof_z1,
            roof_z1 + 0.045,
            color1,
            0.82
        )

    # Pickup bed
    if "pickup" in body:

        add_box(
            -length * 0.48,
            -length * 0.08,
            -width * 0.47,
            width * 0.47,
            ground + base_h * 0.53,
            ground + base_h * 0.72,
            color2,
            0.82
        )

    # Front splitter
    add_box(
        length * 0.44,
        length * 0.505,
        -width * 0.44,
        width * 0.44,
        ground + 0.04,
        ground + 0.13,
        "#0C0E11"
    )

    # Rear diffuser
    add_box(
        -length * 0.505,
        -length * 0.44,
        -width * 0.44,
        width * 0.44,
        ground + 0.04,
        ground + 0.12,
        "#0C0E11"
    )

    # Wheels
    wheel_positions = [
        (length * 0.36, width * 0.51),
        (length * 0.36, -width * 0.51),
        (-length * 0.36, width * 0.51),
        (-length * 0.36, -width * 0.51),
    ]

    for wx, wy in wheel_positions:

        angle = np.linspace(
            0,
            2 * np.pi,
            48
        )

        fig.add_trace(
            go.Scatter3d(
                x=wx + wheel_r * 0.92 * np.cos(angle),
                y=np.full_like(angle, wy),
                z=(
                    ground
                    + wheel_r
                    + wheel_r * 0.92 * np.sin(angle)
                ),
                mode="lines",
                line=dict(
                    color="#050607",
                    width=14
                ),
                showlegend=False,
                hoverinfo="skip"
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[wx],
                y=[wy],
                z=[ground + wheel_r],
                mode="markers",
                marker=dict(
                    size=9,
                    color="#8E969E",
                    line=dict(
                        color="#111418",
                        width=2
                    )
                ),
                showlegend=False,
                hoverinfo="skip"
            )
        )

    # Lights
    lamp_z = (
        ground
        + base_h * 0.38
    )

    fig.add_trace(
        go.Scatter3d(
            x=[
                length * 0.495,
                length * 0.495
            ],
            y=[
                -width * 0.30,
                width * 0.30
            ],
            z=[
                lamp_z,
                lamp_z
            ],
            mode="markers",
            marker=dict(
                size=7,
                color=COLORS["telemetry"]
            ),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[
                -length * 0.495,
                -length * 0.495
            ],
            y=[
                -width * 0.30,
                width * 0.30
            ],
            z=[
                lamp_z,
                lamp_z
            ],
            mode="markers",
            marker=dict(
                size=6,
                color=COLORS["redline"]
            ),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    if sporty:

        fig.add_trace(
            go.Scatter3d(
                x=[
                    length * 0.20,
                    length * 0.20
                ],
                y=[
                    -width * 0.525,
                    width * 0.525
                ],
                z=[
                    ground + base_h * 0.28,
                    ground + base_h * 0.28
                ],
                mode="markers",
                marker=dict(
                    size=5,
                    color=COLORS["accent"]
                ),
                showlegend=False,
                hoverinfo="skip"
            )
        )

    # --------------------------------------------------------
    # Axis ranges sized to the car itself (not a generic square).
    # Previously all three axes shared one "max_dim" derived from
    # length/width, so the z-axis range was ~3x taller than the
    # actual car — the mesh ended up occupying a small sliver at
    # the bottom of a much taller invisible box, rendering as a
    # tiny, oddly-proportioned shape. Fit each axis to what's
    # actually there, with a little breathing room.
    # --------------------------------------------------------

    car_top = (
        ground
        + base_h * 1.15
        + 0.05
    )

    x_span = length / 2 + 0.35
    y_span = width / 2 + 0.35
    z_span = max(car_top, 0.9)

    fig.update_layout(
        height=height_px,
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        scene=dict(
            xaxis=dict(
                visible=False,
                range=[
                    -x_span,
                    x_span
                ]
            ),
            yaxis=dict(
                visible=False,
                range=[
                    -y_span,
                    y_span
                ]
            ),
            zaxis=dict(
                visible=False,
                range=[
                    0,
                    z_span
                ]
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(
                    x=1.65,
                    y=1.55,
                    z=0.95
                ),
                center=dict(
                    x=0,
                    y=0,
                    z=-0.05
                )
            ),
            bgcolor="rgba(0,0,0,0)"
        )
    )

    return fig


# ============================================================
# VEHICLE OPTIONS
# ============================================================

def get_vehicle_options(dataset):

    # --------------------------------------------
    # NORMAL CASE
    # --------------------------------------------

    if (
        "Manufacturer" in dataset.columns
        and "Brand" in dataset.columns
    ):

        temp = dataset[
            ["Manufacturer", "Brand"]
        ].copy()

        temp["Manufacturer"] = (
            temp["Manufacturer"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        temp["Brand"] = (
            temp["Brand"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        temp = temp[
            (
                temp["Manufacturer"] != ""
            )
            & (
                temp["Brand"] != ""
            )
        ]

        temp = temp.drop_duplicates()

        temp = temp.sort_values(
            ["Manufacturer", "Brand"]
        )

        options = []
        lookup = {}

        for _, row in temp.iterrows():

            manufacturer = row[
                "Manufacturer"
            ]

            brand = row["Brand"]

            label = (
                f"{manufacturer} — {brand}"
            )

            options.append(label)

            lookup[label] = (
                manufacturer,
                brand
            )

        if options:
            return options, lookup

    # --------------------------------------------
    # FALLBACK:
    # ONLY Brand_Manufacturer EXISTS
    # --------------------------------------------

    if (
        "Brand_Manufacturer"
        in dataset.columns
    ):

        combined = (
            dataset[
                "Brand_Manufacturer"
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )

        combined = sorted(
            [
                x for x in
                combined.unique()
                if x
            ]
        )

        lookup = {}

        for value in combined:

            parts = value.split(
                " ",
                1
            )

            if len(parts) == 2:

                lookup[value] = (
                    parts[0],
                    parts[1]
                )

            else:

                lookup[value] = (
                    value,
                    value
                )

        return combined, lookup

    return ["Unknown — Unknown"], {
        "Unknown — Unknown": (
            "Unknown",
            "Unknown"
        )
    }


# ============================================================
# FIND CLOSEST DATASET ROW
# ============================================================

def matching_vehicle_row(
    dataset,
    manufacturer,
    brand,
    year
):

    if (
        "Manufacturer" not in
        dataset.columns
    ):
        return None

    if (
        "Brand" not in
        dataset.columns
    ):
        return None

    pool = dataset[
        (
            dataset["Manufacturer"]
            == manufacturer
        )
        &
        (
            dataset["Brand"]
            == brand
        )
    ].copy()

    if pool.empty:
        return None

    years = pd.to_numeric(
        pool["Model Year"],
        errors="coerce"
    )

    exact = pool[
        years == year
    ]

    if not exact.empty:
        return exact.iloc[0]

    pool["_year_diff"] = (
        years - year
    ).abs()

    pool = pool.dropna(
        subset=["_year_diff"]
    )

    if pool.empty:
        return None

    return pool.sort_values(
        "_year_diff"
    ).iloc[0]


# ============================================================
# VEHICLE SELECTOR
# ============================================================

def render_vehicle_selector(
    dataset
):

    values = {}

    # ========================================================
    # VEHICLE IDENTITY
    # ========================================================

    st.markdown(
        "<div class='ap-card'>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='ap-card-title'>"
        "🚗 Vehicle Identity"
        "</div>",
        unsafe_allow_html=True
    )

    options, lookup = (
        get_vehicle_options(
            dataset
        )
    )

    if not options:

        options = [
            "Unknown — Unknown"
        ]

        lookup = {
            "Unknown — Unknown":
            (
                "Unknown",
                "Unknown"
            )
        }

    preferred = (
        "Mercedes-Benz — A 35 AMG"
    )

    default_index = (
        options.index(preferred)
        if preferred in options
        else 0
    )

    selected = st.selectbox(
        "Choose Your Car",
        options,
        index=default_index,
        key="vehicle_identity",
        help=(
            "Click and type to search "
            "for your manufacturer/model."
        )
    )

    manufacturer, brand = lookup.get(
        selected,
        (
            "Unknown",
            selected
        )
    )

    # --------------------------------------------------------
    # YEAR 1950–2026
    # --------------------------------------------------------

    model_year = st.number_input(
        "Model Year",
        min_value=1950,
        max_value=2026,
        value=2026,
        step=1,
        key="vehicle_year",
        help=(
            "Supported range: "
            "1950 to 2026."
        )
    )

    model_year = int(
        np.clip(
            model_year,
            1950,
            2026
        )
    )

    values["Manufacturer"] = (
        manufacturer
    )

    values["Brand"] = brand

    values["Brand_Manufacturer"] = (
        f"{manufacturer} {brand}"
    ).strip()

    values["Model Year"] = (
        model_year
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    selected_row = (
        matching_vehicle_row(
            dataset,
            manufacturer,
            brand,
            model_year
        )
    )

    # ========================================================
    # BODY / DRIVETRAIN
    # ========================================================

    st.markdown(
        "<div class='ap-card'>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='ap-card-title'>"
        "⚙️ Vehicle Configuration"
        "</div>",
        unsafe_allow_html=True
    )

    body_types = safe_unique(
        dataset,
        "Body Type",
        [
            "Sedan",
            "SUV",
            "Coupe",
            "Hatchback",
            "Convertible",
            "Pickup",
            "Wagon",
            "Van"
        ]
    )

    origins = safe_unique(
        dataset,
        "Origin Country",
        ["Unknown"]
    )

    additional_types = safe_unique(
        dataset,
        "Additional Type",
        ["Standard"]
    )

    gear_types = safe_unique(
        dataset,
        "gear_type",
        ["A"]
    )

    if "gear_count" in dataset.columns:

        gear_counts = (
            pd.to_numeric(
                dataset["gear_count"],
                errors="coerce"
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        gear_counts = sorted(
            gear_counts
        )

    else:

        gear_counts = [
            5,
            6,
            7,
            8
        ]

    if not gear_counts:
        gear_counts = [6]

    # --------------------------------------------------------
    # Defaults
    # --------------------------------------------------------

    if selected_row is not None:

        body_default = str(
            safe_value(
                selected_row.get(
                    "Body Type"
                ),
                body_types[0]
            )
        )

        origin_default = str(
            safe_value(
                selected_row.get(
                    "Origin Country"
                ),
                origins[0]
            )
        )

        add_default = str(
            safe_value(
                selected_row.get(
                    "Additional Type"
                ),
                additional_types[0]
            )
        )

        gear_default = str(
            safe_value(
                selected_row.get(
                    "gear_type"
                ),
                gear_types[0]
            )
        )

        gear_count_default = int(
            safe_value(
                selected_row.get(
                    "gear_count"
                ),
                gear_counts[
                    len(gear_counts) // 2
                ]
            )
        )

    else:

        body_default = body_types[0]
        origin_default = origins[0]
        add_default = additional_types[0]
        gear_default = gear_types[0]
        gear_count_default = gear_counts[
            len(gear_counts) // 2
        ]

    # --------------------------------------------------------
    # BODY TYPE — EASY SEARCH
    # --------------------------------------------------------

    body_index = (
        body_types.index(
            body_default
        )
        if body_default
        in body_types
        else 0
    )

    body_type = st.selectbox(
        "🚗 Body Type",
        body_types,
        index=body_index,
        key="body_type",
        format_func=lambda x:
            f"{BODY_TYPE_ICONS.get(x, '🚗')} {x}",
    )

    c1, c2 = st.columns(2)

    with c1:

        gear_index = (
            gear_types.index(
                gear_default
            )
            if gear_default
            in gear_types
            else 0
        )

        gear_type = st.selectbox(
            "Transmission",
            gear_types,
            index=gear_index,
            key="gear_type"
        )

        st.caption(
            GEAR_TYPE_LABELS.get(
                gear_type,
                gear_type
            )
        )

    with c2:

        gear_count_index = (
            gear_counts.index(
                gear_count_default
            )
            if gear_count_default
            in gear_counts
            else len(gear_counts) // 2
        )

        gear_count = st.selectbox(
            "Gears",
            gear_counts,
            index=gear_count_index,
            key="gear_count"
        )

    c1, c2 = st.columns(2)

    with c1:

        origin_index = (
            origins.index(
                origin_default
            )
            if origin_default
            in origins
            else 0
        )

        origin = st.selectbox(
            "Origin Country",
            origins,
            index=origin_index,
            key="origin_country"
        )

    with c2:

        add_index = (
            additional_types.index(
                add_default
            )
            if add_default
            in additional_types
            else 0
        )

        additional_type = st.selectbox(
            "Additional Type",
            additional_types,
            index=add_index,
            key="additional_type"
        )

    values.update(
        {
            "Body Type": body_type,
            "Origin Country": origin,
            "Additional Type": additional_type,
            "gear_type": gear_type,
            "gear_count": float(
                gear_count
            ),
        }
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.markdown(
        "<div class='ap-card'>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='ap-card-title'>"
        "🏁 Performance & Efficiency"
        "</div>",
        unsafe_allow_html=True
    )

    def row_value(
        column,
        fallback
    ):

        if selected_row is None:
            return fallback

        return safe_value(
            selected_row.get(
                column
            ),
            fallback
        )

    # --------------------------------------------------------
    # Ranges
    # --------------------------------------------------------

    weight_lo, weight_hi = (
        safe_numeric_range(
            dataset["Weight"],
            (900, 3200)
        )
    )

    perf_lo, perf_hi = (
        safe_numeric_range(
            dataset[
                "Performance 0-100 kph (sec)"
            ],
            (2, 20)
        )
    )

    top_lo, top_hi = (
        safe_numeric_range(
            dataset[
                "Top speed (kph)"
            ],
            (120, 350)
        )
    )

    fuel_lo, fuel_hi = (
        safe_numeric_range(
            dataset[
                "Fuel Econ (L/100km)"
            ],
            (3, 20)
        )
    )

    cost_lo, cost_hi = (
        safe_numeric_range(
            dataset[
                "Approx Cost"
            ],
            (50000, 2000000)
        )
    )

    # --------------------------------------------------------
    # Defaults
    # --------------------------------------------------------

    default_weight = float(
        np.clip(
            float(
                row_value(
                    "Weight",
                    dataset[
                        "Weight"
                    ].median()
                )
            ),
            weight_lo,
            weight_hi
        )
    )

    default_perf = float(
        np.clip(
            float(
                row_value(
                    "Performance 0-100 kph (sec)",
                    dataset[
                        "Performance 0-100 kph (sec)"
                    ].median()
                )
            ),
            perf_lo,
            perf_hi
        )
    )

    default_top = float(
        np.clip(
            float(
                row_value(
                    "Top speed (kph)",
                    dataset[
                        "Top speed (kph)"
                    ].median()
                )
            ),
            top_lo,
            top_hi
        )
    )

    default_fuel = float(
        np.clip(
            float(
                row_value(
                    "Fuel Econ (L/100km)",
                    dataset[
                        "Fuel Econ (L/100km)"
                    ].median()
                )
            ),
            fuel_lo,
            fuel_hi
        )
    )

    default_cost = float(
        np.clip(
            float(
                row_value(
                    "Approx Cost",
                    dataset[
                        "Approx Cost"
                    ].median()
                )
            ),
            cost_lo,
            cost_hi
        )
    )

    c1, c2 = st.columns(2)

    with c1:

        weight = st.number_input(
            "Weight (kg)",
            min_value=float(
                round(weight_lo)
            ),
            max_value=float(
                round(weight_hi)
            ),
            value=float(
                round(default_weight)
            ),
            step=10.0,
            key="input_weight"
        )

        perf = st.number_input(
            "0–100 km/h (sec)",
            min_value=float(
                round(
                    perf_lo,
                    1
                )
            ),
            max_value=float(
                round(
                    perf_hi,
                    1
                )
            ),
            value=float(
                round(
                    default_perf,
                    1
                )
            ),
            step=0.1,
            key="input_perf"
        )

        top_speed = st.number_input(
            "Top Speed (km/h)",
            min_value=float(
                round(top_lo)
            ),
            max_value=float(
                round(top_hi)
            ),
            value=float(
                round(default_top)
            ),
            step=1.0,
            key="input_top"
        )

    with c2:

        fuel_l100 = st.number_input(
            "Fuel Economy (L/100km)",
            min_value=float(
                round(
                    fuel_lo,
                    1
                )
            ),
            max_value=float(
                round(
                    fuel_hi,
                    1
                )
            ),
            value=float(
                round(
                    default_fuel,
                    1
                )
            ),
            step=0.1,
            key="input_fuel"
        )

        fuel_kml = (
            100 / fuel_l100
            if fuel_l100 > 0
            else 0
        )

        st.caption(
            f"≈ **{fuel_kml:.1f} km/L**"
        )

        approx_cost = st.number_input(
            "Approx Cost (AED)",
            min_value=float(
                round(
                    cost_lo,
                    -3
                )
            ),
            max_value=float(
                round(
                    cost_hi,
                    -3
                )
            ),
            value=float(
                round(
                    default_cost,
                    -3
                )
            ),
            step=1000.0,
            key="input_cost"
        )

    values.update(
        {
            "Weight": weight,
            "Performance 0-100 kph (sec)": perf,
            "Top speed (kph)": top_speed,
            "Fuel Econ (L/100km)": fuel_l100,
            "Fuel Econ (km/L)": fuel_kml,
            "Approx Cost": approx_cost,
        }
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    return values


# ============================================================
# VEHICLE PREVIEW
# ============================================================

def render_vehicle_preview(
    values,
    dataset
):

    st.markdown(
        "<div class='ap-card'>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='ap-card-title'>"
        "👁️ Vehicle Preview"
        "</div>",
        unsafe_allow_html=True
    )

    fig = render_vehicle_visual(
        values["Manufacturer"],
        values["Brand"],
        values["Body Type"],
        values["Weight"],
        values["Top speed (kph)"],
        values[
            "Performance 0-100 kph (sec)"
        ],
        values["gear_type"],
        height_px=320
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )

    st.markdown(
        f"""
        <div style="text-align:center">

            <div class="ap-display"
                 style="font-size:1.7rem">

                {values["Manufacturer"]}
                {values["Brand"]}

            </div>

            <div style="
                color:{COLORS["text_dim"]};
                margin-top:0.3rem;
            ">

                {BODY_TYPE_ICONS.get(
                    values["Body Type"],
                    "🚗"
                )}

                {values["Body Type"]}
                ·
                {values["Model Year"]}

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<hr class='ap-div'>",
        unsafe_allow_html=True
    )

    specs = [
        (
            "Body Type",
            values["Body Type"]
        ),
        (
            "Transmission",
            GEAR_TYPE_LABELS.get(
                values["gear_type"],
                values["gear_type"]
            )
        ),
        (
            "Gears",
            int(values["gear_count"])
        ),
        (
            "Weight",
            f'{values["Weight"]:,.0f} kg'
        ),
        (
            "0–100 km/h",
            f'{values["Performance 0-100 kph (sec)"]:.1f} s'
        ),
        (
            "Top Speed",
            f'{values["Top speed (kph)"]:.0f} km/h'
        ),
        (
            "Fuel",
            f'{values["Fuel Econ (L/100km)"]:.1f} L/100km'
        ),
    ]

    for label, value in specs:

        st.markdown(
            f"""
            <div class="ap-spec-row">
                <span class="k">
                    {label}
                </span>
                <span class="v">
                    {value}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# POWER GAUGE
# ============================================================

def render_power_gauge(
    hp,
    max_scale
):

    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=hp,
            gauge={
                "axis": {
                    "range": [
                        0,
                        max_scale
                    ],
                    "tickcolor":
                        COLORS["text_dim"]
                },
                "bar": {
                    "color":
                        COLORS["accent"]
                },
                "bgcolor":
                    "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [
                            0,
                            max_scale * 0.5
                        ],
                        "color":
                            "rgba(47,212,192,0.18)"
                    },
                    {
                        "range": [
                            max_scale * 0.5,
                            max_scale * 0.8
                        ],
                        "color":
                            "rgba(255,90,31,0.20)"
                    },
                    {
                        "range": [
                            max_scale * 0.8,
                            max_scale
                        ],
                        "color":
                            "rgba(255,45,85,0.28)"
                    }
                ],
                "threshold": {
                    "line": {
                        "color":
                            COLORS["redline"],
                        "width": 3
                    },
                    "thickness": 0.85,
                    "value": hp
                }
            }
        )
    )

    fig.update_layout(
        height=220,
        margin=dict(
            l=20,
            r=20,
            t=10,
            b=10
        ),
        paper_bgcolor=
            "rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# ============================================================
# PREDICTION RESULT
# ============================================================

def render_prediction_result(
    hp,
    results_df,
    dataset,
    values
):

    hp = max(
        float(hp),
        0
    )

    kw = hp * 0.7457

    tier_label, tier_emoji = (
        classify_performance(
            hp
        )
    )

    if values["Weight"]:

        ptw = (
            hp
            / values["Weight"]
            * 1000
        )

    else:

        ptw = np.nan

    st.markdown(
        "<div class='ap-card'>",
        unsafe_allow_html=True
    )

    fig = render_vehicle_visual(
        values["Manufacturer"],
        values["Brand"],
        values["Body Type"],
        values["Weight"],
        values["Top speed (kph)"],
        values[
            "Performance 0-100 kph (sec)"
        ],
        values["gear_type"],
        height_px=250
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )

    st.markdown(
        f"""
        <div class="ap-result">

            <div class="label">
                Predicted Power
            </div>

            <div class="hp">
                {hp:,.0f}
            </div>

            <div class="kw">
                Horsepower · ≈ {kw:,.0f} kW
            </div>

            <div class="ap-tier">
                {tier_emoji}
                {tier_label}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    max_scale = max(
        800,
        float(
            dataset[
                "Power (hp)"
            ].max()
        ) * 1.05
    )

    render_power_gauge(
        hp,
        max_scale
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            f"""
            <div class="ap-metric">
                <div class="v">
                    {ptw:.1f}
                </div>
                <div class="l">
                    HP per Tonne
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="ap-metric">
                <div class="v">
                    {values["Model Year"]}
                </div>
                <div class="l">
                    Model Year
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    if (
        results_df is not None
        and len(results_df) > 0
    ):

        columns = set(
            results_df.columns
        )

        if {
            "Test Accuracy",
            "MAE",
            "RMSE"
        }.issubset(columns):

            try:

                best = (
                    results_df
                    .sort_values(
                        "Test Accuracy",
                        ascending=False
                    )
                    .iloc[0]
                )

                c1, c2, c3 = (
                    st.columns(3)
                )

                with c1:

                    st.metric(
                        "Model R²",
                        f'{best["Test Accuracy"]:.1f}%'
                    )

                with c2:

                    st.metric(
                        "MAE",
                        f'{best["MAE"]:.0f} HP'
                    )

                with c3:

                    st.metric(
                        "RMSE",
                        f'{best["RMSE"]:.0f} HP'
                    )

            except Exception:
                pass

    st.markdown(
        """
        <div class="ap-disclaimer">

        This is an ML-based estimate.
        It is a point prediction and may differ
        from manufacturer specifications or
        real-world measurements.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CAR BATTLE
# ============================================================

def render_battle_car_config(
    dataset,
    side_label,
    key_prefix
):
    """
    Full component editor for one side of a battle — same level
    of control as the main Predictor tab (identity, drivetrain,
    performance), just condensed. Returns a values dict compatible
    with build_input_dataframe / predict_power.
    """

    values = {}

    options, lookup = get_vehicle_options(dataset)

    if not options:
        options = ["Unknown — Unknown"]
        lookup = {"Unknown — Unknown": ("Unknown", "Unknown")}

    default_index = 0 if side_label == "A" else min(1, len(options) - 1)

    selected = st.selectbox(
        f"Car {side_label}",
        options,
        index=default_index,
        key=f"{key_prefix}_identity"
    )

    manufacturer, brand = lookup.get(
        selected,
        ("Unknown", selected)
    )

    model_year = st.number_input(
        "Model Year",
        min_value=1950,
        max_value=2026,
        value=2026,
        step=1,
        key=f"{key_prefix}_year"
    )

    model_year = int(np.clip(model_year, 1950, 2026))

    values["Manufacturer"] = manufacturer
    values["Brand"] = brand
    values["Brand_Manufacturer"] = f"{manufacturer} {brand}".strip()
    values["Model Year"] = model_year

    selected_row = matching_vehicle_row(
        dataset,
        manufacturer,
        brand,
        model_year
    )

    body_types = safe_unique(
        dataset,
        "Body Type",
        ["Sedan", "SUV", "Coupe", "Hatchback", "Convertible", "Pickup", "Wagon", "Van"]
    )

    origins = safe_unique(dataset, "Origin Country", ["Unknown"])
    additional_types = safe_unique(dataset, "Additional Type", ["Standard"])
    gear_types = safe_unique(dataset, "gear_type", ["A"])

    if "gear_count" in dataset.columns:

        gear_counts = sorted(
            pd.to_numeric(dataset["gear_count"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

    else:
        gear_counts = [5, 6, 7, 8]

    if not gear_counts:
        gear_counts = [6]

    def default_of(column, fallback_list):

        if selected_row is not None:

            val = str(safe_value(selected_row.get(column), fallback_list[0]))

            if val in fallback_list:
                return val

        return fallback_list[0]

    body_default = default_of("Body Type", body_types)
    origin_default = default_of("Origin Country", origins)
    add_default = default_of("Additional Type", additional_types)
    gear_default = default_of("gear_type", gear_types)

    if selected_row is not None:

        gear_count_default = int(
            safe_value(
                selected_row.get("gear_count"),
                gear_counts[len(gear_counts) // 2]
            )
        )

        if gear_count_default not in gear_counts:
            gear_count_default = gear_counts[len(gear_counts) // 2]

    else:
        gear_count_default = gear_counts[len(gear_counts) // 2]

    body_type = st.selectbox(
        "Body Type",
        body_types,
        index=body_types.index(body_default) if body_default in body_types else 0,
        key=f"{key_prefix}_body",
        format_func=lambda x: f"{BODY_TYPE_ICONS.get(x, '🚗')} {x}",
    )

    gc1, gc2 = st.columns(2)

    with gc1:

        gear_type = st.selectbox(
            "Transmission",
            gear_types,
            index=gear_types.index(gear_default) if gear_default in gear_types else 0,
            key=f"{key_prefix}_gear"
        )

    with gc2:

        gear_count = st.selectbox(
            "Gears",
            gear_counts,
            index=gear_counts.index(gear_count_default) if gear_count_default in gear_counts else 0,
            key=f"{key_prefix}_gearcount"
        )

    oc1, oc2 = st.columns(2)

    with oc1:

        origin = st.selectbox(
            "Origin",
            origins,
            index=origins.index(origin_default) if origin_default in origins else 0,
            key=f"{key_prefix}_origin"
        )

    with oc2:

        additional_type = st.selectbox(
            "Type",
            additional_types,
            index=additional_types.index(add_default) if add_default in additional_types else 0,
            key=f"{key_prefix}_addl"
        )

    values.update({
        "Body Type": body_type,
        "Origin Country": origin,
        "Additional Type": additional_type,
        "gear_type": gear_type,
        "gear_count": float(gear_count),
    })

    def row_value(column, fallback):

        if selected_row is None:
            return fallback

        return safe_value(selected_row.get(column), fallback)

    weight_lo, weight_hi = safe_numeric_range(dataset["Weight"], (900, 3200))
    perf_lo, perf_hi = safe_numeric_range(dataset["Performance 0-100 kph (sec)"], (2, 20))
    top_lo, top_hi = safe_numeric_range(dataset["Top speed (kph)"], (120, 350))
    fuel_lo, fuel_hi = safe_numeric_range(dataset["Fuel Econ (L/100km)"], (3, 20))
    cost_lo, cost_hi = safe_numeric_range(dataset["Approx Cost"], (50000, 2000000))

    default_weight = float(np.clip(float(row_value("Weight", dataset["Weight"].median())), weight_lo, weight_hi))
    default_perf = float(np.clip(float(row_value("Performance 0-100 kph (sec)", dataset["Performance 0-100 kph (sec)"].median())), perf_lo, perf_hi))
    default_top = float(np.clip(float(row_value("Top speed (kph)", dataset["Top speed (kph)"].median())), top_lo, top_hi))
    default_fuel = float(np.clip(float(row_value("Fuel Econ (L/100km)", dataset["Fuel Econ (L/100km)"].median())), fuel_lo, fuel_hi))
    default_cost = float(np.clip(float(row_value("Approx Cost", dataset["Approx Cost"].median())), cost_lo, cost_hi))

    pc1, pc2 = st.columns(2)

    with pc1:

        weight = st.number_input(
            "Weight (kg)",
            min_value=float(round(weight_lo)),
            max_value=float(round(weight_hi)),
            value=float(round(default_weight)),
            step=10.0,
            key=f"{key_prefix}_weight"
        )

        perf = st.number_input(
            "0–100 km/h (s)",
            min_value=float(round(perf_lo, 1)),
            max_value=float(round(perf_hi, 1)),
            value=float(round(default_perf, 1)),
            step=0.1,
            key=f"{key_prefix}_perf"
        )

        top_speed = st.number_input(
            "Top Speed (km/h)",
            min_value=float(round(top_lo)),
            max_value=float(round(top_hi)),
            value=float(round(default_top)),
            step=1.0,
            key=f"{key_prefix}_top"
        )

    with pc2:

        fuel_l100 = st.number_input(
            "Fuel (L/100km)",
            min_value=float(round(fuel_lo, 1)),
            max_value=float(round(fuel_hi, 1)),
            value=float(round(default_fuel, 1)),
            step=0.1,
            key=f"{key_prefix}_fuel"
        )

        approx_cost = st.number_input(
            "Cost (AED)",
            min_value=float(round(cost_lo, -3)),
            max_value=float(round(cost_hi, -3)),
            value=float(round(default_cost, -3)),
            step=1000.0,
            key=f"{key_prefix}_cost"
        )

    fuel_kml = 100 / fuel_l100 if fuel_l100 > 0 else 0

    values.update({
        "Weight": weight,
        "Performance 0-100 kph (sec)": perf,
        "Top speed (kph)": top_speed,
        "Fuel Econ (L/100km)": fuel_l100,
        "Fuel Econ (km/L)": fuel_kml,
        "Approx Cost": approx_cost,
    })

    return values


def battle_metric_value(values, hp, column):

    if column == "Power (hp)":
        return float(hp)

    return float(values.get(column, 0.0))


def render_battle_tab(dataset, model):

    st.markdown("### ⚔️ Car Battle")

    st.caption(
        "Configure two cars component by component — the trained "
        "model predicts each one's horsepower, then they go head to head."
    )

    col_a, col_b = st.columns(2)

    with col_a:

        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>🅰️ Contender A</div>", unsafe_allow_html=True)
        values_a = render_battle_car_config(dataset, "A", "battle_a")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:

        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>🅱️ Contender B</div>", unsafe_allow_html=True)
        values_b = render_battle_car_config(dataset, "B", "battle_b")
        st.markdown("</div>", unsafe_allow_html=True)

    if not st.button(
        "⚔️ START BATTLE",
        use_container_width=True,
        key="battle_button"
    ):
        return

    try:
        hp_a = predict_power(model, build_input_dataframe(dict(values_a)))
        hp_b = predict_power(model, build_input_dataframe(dict(values_b)))
    except Exception as error:
        st.error("❌ Prediction failed.")
        with st.expander("Technical details"):
            st.code(str(error))
        return

    hp_a = max(hp_a, 0.0)
    hp_b = max(hp_b, 0.0)

    preview_a, preview_b = st.columns(2)

    with preview_a:

        fig_a = render_vehicle_visual(
            values_a["Manufacturer"], values_a["Brand"], values_a["Body Type"],
            values_a["Weight"], values_a["Top speed (kph)"],
            values_a["Performance 0-100 kph (sec)"], values_a["gear_type"],
            height_px=220,
        )

        st.plotly_chart(fig_a, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"<div class='ap-display' style='text-align:center;font-size:1.3rem'>{values_a['Manufacturer']} {values_a['Brand']}</div>",
            unsafe_allow_html=True
        )

    with preview_b:

        fig_b = render_vehicle_visual(
            values_b["Manufacturer"], values_b["Brand"], values_b["Body Type"],
            values_b["Weight"], values_b["Top speed (kph)"],
            values_b["Performance 0-100 kph (sec)"], values_b["gear_type"],
            height_px=220,
        )

        st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"<div class='ap-display' style='text-align:center;font-size:1.3rem'>{values_b['Manufacturer']} {values_b['Brand']}</div>",
            unsafe_allow_html=True
        )

    score_a = 0
    score_b = 0

    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>📊 Head-to-Head</div>", unsafe_allow_html=True)

    for column, direction, unit in COMPARISON_METRICS:

        val_a = battle_metric_value(values_a, hp_a, column)
        val_b = battle_metric_value(values_b, hp_b, column)

        if direction == "higher":
            a_wins = val_a > val_b
            b_wins = val_b > val_a
        elif direction == "lower":
            a_wins = val_a < val_b
            b_wins = val_b < val_a
        else:
            a_wins = False
            b_wins = False

        if a_wins:
            score_a += 1
        if b_wins:
            score_b += 1

        style_a = f"color:{COLORS['accent']};font-weight:700;" if a_wins else f"color:{COLORS['text']};"
        style_b = f"color:{COLORS['accent']};font-weight:700;" if b_wins else f"color:{COLORS['text']};"

        st.markdown(
            f"<div class='ap-spec-row'>"
            f"<span class='v' style='{style_a}'>{val_a:,.1f} {unit}</span>"
            f"<span class='k'>{column}</span>"
            f"<span class='v' style='{style_b}'>{val_b:,.1f} {unit}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if score_a > score_b:
        winner = f"{values_a['Manufacturer']} {values_a['Brand']}"
    elif score_b > score_a:
        winner = f"{values_b['Manufacturer']} {values_b['Brand']}"
    else:
        winner = "Tie"

    st.markdown(
        f"<div class='ap-winner-banner'>"
        f"<div class='ap-eyebrow'>Battle Result</div>"
        f"<div class='t'>🏆 {winner}</div>"
        f"<div style='color:{COLORS['text_dim']}'>{score_a} - {score_b}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# DATASET EXPLORER
# ============================================================

def numeric_columns(
    dataset
):

    result = []

    for column in (
        NUMERIC_COLS
        + ["Power (hp)"]
    ):

        if column in dataset.columns:

            if (
                pd.to_numeric(
                    dataset[column],
                    errors="coerce"
                )
                .notna()
                .any()
            ):

                result.append(column)

    return result


def categorical_columns(
    dataset
):

    candidates = [
        "Manufacturer",
        "Brand",
        "Body Type",
        "Origin Country",
        "Additional Type",
        "gear_type",
        "Brand_Manufacturer",
    ]

    return [
        x
        for x in candidates
        if x in dataset.columns
    ]


def apply_filters(
    dataset
):

    with st.expander(
        "🔍 Advanced Filters",
        expanded=False
    ):

        c1, c2, c3 = (
            st.columns(3)
        )

        with c1:

            manufacturers = safe_unique(
                dataset,
                "Manufacturer"
            )

            selected_makers = (
                st.multiselect(
                    "Manufacturer",
                    manufacturers,
                    key="filter_makers"
                )
            )

            bodies = safe_unique(
                dataset,
                "Body Type"
            )

            selected_bodies = (
                st.multiselect(
                    "Body Type",
                    bodies,
                    key="filter_body"
                )
            )

        with c2:

            origins = safe_unique(
                dataset,
                "Origin Country"
            )

            selected_origins = (
                st.multiselect(
                    "Origin Country",
                    origins,
                    key="filter_origin"
                )
            )

            gears = safe_unique(
                dataset,
                "gear_type"
            )

            selected_gears = (
                st.multiselect(
                    "Transmission",
                    gears,
                    key="filter_gear"
                )
            )

        with c3:

            start_year = st.number_input(
                "From Year",
                min_value=1950,
                max_value=2026,
                value=1950,
                step=1,
                key="filter_year_from"
            )

            end_year = st.number_input(
                "To Year",
                min_value=1950,
                max_value=2026,
                value=2026,
                step=1,
                key="filter_year_to"
            )

    result = dataset.copy()

    if selected_makers:

        result = result[
            result[
                "Manufacturer"
            ].isin(
                selected_makers
            )
        ]

    if selected_bodies:

        result = result[
            result[
                "Body Type"
            ].isin(
                selected_bodies
            )
        ]

    if selected_origins:

        result = result[
            result[
                "Origin Country"
            ].isin(
                selected_origins
            )
        ]

    if selected_gears:

        result = result[
            result[
                "gear_type"
            ].isin(
                selected_gears
            )
        ]

    lo = min(
        start_year,
        end_year
    )

    hi = max(
        start_year,
        end_year
    )

    result = result[
        (
            result[
                "Model Year"
            ] >= lo
        )
        &
        (
            result[
                "Model Year"
            ] <= hi
        )
    ]

    st.caption(
        f"**{len(result):,}** "
        f"of **{len(dataset):,}** "
        "vehicles match."
    )

    return result


def viz_overview(
    df
):

    if df.empty:

        st.info(
            "No vehicles match."
        )

        return

    c1, c2, c3, c4, c5 = (
        st.columns(5)
    )

    metrics = [
        (
            "Cars",
            f"{len(df):,}"
        ),
        (
            "Manufacturers",
            f"{df['Manufacturer'].nunique():,}"
        ),
        (
            "Brands",
            f"{df['Brand'].nunique():,}"
        ),
        (
            "Average HP",
            f"{df['Power (hp)'].mean():,.0f}"
        ),
        (
            "Maximum HP",
            f"{df['Power (hp)'].max():,.0f}"
        ),
    ]

    for column, (
        label,
        value
    ) in zip(
        [c1,c2,c3,c4,c5],
        metrics
    ):

        with column:

            st.markdown(
                f"""
                <div class="ap-metric">

                    <div class="v">
                        {value}
                    </div>

                    <div class="l">
                        {label}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    fig = px.histogram(
        df,
        x="Power (hp)",
        nbins=40,
        color_discrete_sequence=[
            COLORS["accent"]
        ]
    )

    fig.update_layout(
        paper_bgcolor=
            "rgba(0,0,0,0)",
        plot_bgcolor=
            "rgba(0,0,0,0)",
        font={
            "color":
                COLORS["text"]
        },
        height=350
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


def viz_scatter(
    df
):

    cols = numeric_columns(
        df
    )

    if len(cols) < 2:

        st.info(
            "Not enough numeric data."
        )

        return

    c1, c2 = (
        st.columns(2)
    )

    with c1:

        x = st.selectbox(
            "X Axis",
            cols,
            index=(
                cols.index("Weight")
                if "Weight" in cols
                else 0
            )
        )

    with c2:

        y = st.selectbox(
            "Y Axis",
            cols,
            index=(
                cols.index("Power (hp)")
                if "Power (hp)" in cols
                else 1
            )
        )

    sub = df[
        [x, y]
    ].dropna()

    if sub.empty:

        st.info(
            "No data available."
        )

        return

    fig = px.scatter(
        sub,
        x=x,
        y=y,
        color_discrete_sequence=[
            COLORS["accent"]
        ]
    )

    fig.update_layout(
        paper_bgcolor=
            "rgba(0,0,0,0)",
        plot_bgcolor=
            "rgba(0,0,0,0)",
        font={
            "color":
                COLORS["text"]
        },
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


def viz_correlation(
    df
):

    cols = numeric_columns(
        df
    )

    if len(cols) < 2:

        st.info(
            "Not enough numeric features."
        )

        return

    corr = (
        df[cols]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .corr()
    )

    threshold = st.select_slider(
        "Minimum absolute correlation",
        options=[
            0,
            0.5,
            0.7,
            0.8,
            0.9,
            0.95
        ],
        value=0.0
    )

    display = corr.copy()

    if threshold > 0:

        display = display.where(
            display.abs()
            >= threshold
        )

    fig = px.imshow(
        display,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto"
    )

    fig.update_layout(
        paper_bgcolor=
            "rgba(0,0,0,0)",
        plot_bgcolor=
            "rgba(0,0,0,0)",
        font={
            "color":
                COLORS["text"]
        },
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


def viz_top_cars(
    df
):

    if df.empty:

        st.info(
            "No data."
        )

        return

    metric = st.selectbox(
        "Rank By",
        [
            "Power (hp)",
            "Top speed (kph)",
            "Performance 0-100 kph (sec)",
            "Fuel Econ (km/L)"
        ]
    )

    top_n = st.selectbox(
        "Top N",
        [5, 10, 20, 50],
        index=1
    )

    ascending = (
        metric
        ==
        "Performance 0-100 kph (sec)"
    )

    top = (
        df
        .dropna(
            subset=[metric]
        )
        .sort_values(
            metric,
            ascending=ascending
        )
        .head(top_n)
    )

    if top.empty:

        st.info(
            "No vehicles available."
        )

        return

    st.dataframe(
        top[
            [
                "Manufacturer",
                "Brand",
                "Model Year",
                "Body Type",
                metric
            ]
        ],
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# MODEL INSIGHTS
# ============================================================

def render_model_insights(
    results_df,
    feature_importance,
    dataset
):

    st.markdown(
        "<div class='ap-card'>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='ap-card-title'>"
        "🧠 Model Insights"
        "</div>",
        unsafe_allow_html=True
    )

    st.write(
        f"Training dataset: "
        f"**{len(dataset):,} vehicles**"
    )

    st.write(
        "Target variable: "
        "**Power (hp)**"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    if (
        results_df is not None
        and not results_df.empty
    ):

        st.dataframe(
            results_df,
            hide_index=True,
            use_container_width=True
        )

    if (
        feature_importance
        is not None
    ):

        try:

            top = (
                feature_importance
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .head(10)
            )

            fig = px.bar(
                top,
                x="Importance",
                y="Feature",
                orientation="h",
                color_discrete_sequence=[
                    COLORS["accent"]
                ]
            )

            fig.update_layout(
                paper_bgcolor=
                    "rgba(0,0,0,0)",
                plot_bgcolor=
                    "rgba(0,0,0,0)",
                font={
                    "color":
                        COLORS["text"]
                },
                height=450
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )

        except Exception:
            pass


# ============================================================
# MAIN
# ============================================================

def main():

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🏎️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    inject_css()

    model_dir = find_model_dir()

    if model_dir is None:

        render_hero()

        st.error(
            "❌ Trained model files are missing."
        )

        st.markdown(
            """
            The application expects the trained
            model artifacts in:

            `/content/models/`

            The required files are:

            - `best_model.joblib`
            - `dataset.joblib`

            The optional analytics files are:

            - `results_df.joblib`
            - `target_col.joblib`
            - `feature_importance.joblib`

            Run the training/export cell once,
            then restart Streamlit.
            """
        )

        st.stop()

    try:

        model = load_model(
            str(model_dir)
        )

        (
            dataset,
            results_df,
            target_col,
            feature_importance
        ) = load_artifacts(
            str(model_dir)
        )

    except Exception as error:

        render_hero()

        st.error(
            "❌ The model or dataset could not be loaded."
        )

        with st.expander(
            "Technical details"
        ):

            st.code(
                str(error)
            )

        st.stop()

    if dataset.empty:

        st.error(
            "The dataset is empty."
        )

        st.stop()

    # --------------------------------------------------------
    # Verify model columns
    # --------------------------------------------------------

    missing_model_columns = [
        column
        for column in REQUIRED_MODEL_COLUMNS
        if column not in dataset.columns
    ]

    # --------------------------------------------------------
    # Automatically rebuild Brand_Manufacturer if necessary
    # --------------------------------------------------------

    if (
        "Brand_Manufacturer"
        not in dataset.columns
    ):

        if (
            "Manufacturer"
            in dataset.columns
            and
            "Brand"
            in dataset.columns
        ):

            dataset[
                "Brand_Manufacturer"
            ] = (
                dataset[
                    "Manufacturer"
                ].astype(str)
                + " "
                + dataset[
                    "Brand"
                ].astype(str)
            ).str.strip()

            missing_model_columns = [
                column
                for column in REQUIRED_MODEL_COLUMNS
                if column not in dataset.columns
            ]

    if missing_model_columns:

        st.error(
            "The saved dataset is missing "
            f"required model columns: "
            f"{missing_model_columns}"
        )

        st.stop()

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    if (
        results_df is not None
        and not results_df.empty
        and "Test Accuracy"
        in results_df.columns
    ):

        try:

            best_name = str(
                results_df
                .sort_values(
                    "Test Accuracy",
                    ascending=False
                )
                .iloc[0]["Model"]
            )

        except Exception:

            best_name = "Extra Trees"

    else:

        best_name = "Extra Trees"

    render_sidebar(
        dataset,
        best_name
    )

    render_hero()

    # ========================================================
    # TABS
    # ========================================================

    tab_predict, tab_battle, tab_explore, tab_insights = (
        st.tabs(
            [
                "🏎️ Predictor",
                "⚔️ Car Battle",
                "🚗 Explore Garage",
                "🧠 Model Insights"
            ]
        )
    )

    # ========================================================
    # PREDICTOR
    # ========================================================

    with tab_predict:

        left, right = (
            st.columns(
                [1.1, 1]
            )
        )

        with left:

            st.markdown(
                "### Configure Your Machine"
            )

            values = (
                render_vehicle_selector(
                    dataset
                )
            )

            predict_clicked = st.button(
                "⚡ PREDICT ENGINE POWER",
                use_container_width=True
            )

        with right:

            st.markdown(
                "### AI Vehicle Preview"
            )

            if predict_clicked:

                try:

                    input_df = (
                        build_input_dataframe(
                            values
                        )
                    )

                    hp = (
                        predict_power(
                            model,
                            input_df
                        )
                    )

                    render_prediction_result(
                        hp,
                        results_df,
                        dataset,
                        values
                    )

                except Exception as error:

                    st.error(
                        "❌ Prediction failed."
                    )

                    with st.expander(
                        "Technical details"
                    ):

                        st.code(
                            str(error)
                        )

            else:

                render_vehicle_preview(
                    values,
                    dataset
                )

    # ========================================================
    # CAR BATTLE
    # ========================================================

    with tab_battle:

        render_battle_tab(dataset, model)

    # ========================================================
    # EXPLORE
    # ========================================================

    with tab_explore:

        st.markdown(
            "### 🚗 Explore the Garage"
        )

        filtered = apply_filters(
            dataset
        )

        visualization = st.selectbox(
            "Visualization",
            [
                "Overview",
                "Scatter Plot",
                "Correlation Heatmap",
                "Top Cars"
            ]
        )

        if visualization == "Overview":

            viz_overview(
                filtered
            )

        elif visualization == "Scatter Plot":

            viz_scatter(
                filtered
            )

        elif visualization == "Correlation Heatmap":

            viz_correlation(
                filtered
            )

        elif visualization == "Top Cars":

            viz_top_cars(
                filtered
            )

    # ========================================================
    # INSIGHTS
    # ========================================================

    with tab_insights:

        render_model_insights(
            results_df,
            feature_importance,
            dataset
        )

    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown(
        "<hr class='ap-div'>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="ap-footer">

        AUTOPOWER AI · Machine Learning
        Automotive Performance Predictor

        <br>

        Python · Scikit-learn · Streamlit · Plotly

        <br>

        © 2026

        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
'''

# ============================================================
# 5. WRITE app.py
# ============================================================

APP_PATH = Path("/content/app.py")

APP_PATH.write_text(
    STREAMLIT_APP_CODE,
    encoding="utf-8"
)

print()
print("============================================================")
print("✅ AUTOPOWER AI APP CREATED")
print("============================================================")
print()
print("App:")
print(APP_PATH)
print()
print("Model directory:")
print(MODEL_DIR)
print()
print("Files:")
for file in sorted(MODEL_DIR.iterdir()):
    print(" ✓", file.name)

# ============================================================
# 6. START STREAMLIT
# ============================================================

print()
print("Starting Streamlit...")
print()

# Stop previous Streamlit processes if present
os.system(
    "pkill -f 'streamlit run /content/app.py' "
    ">/dev/null 2>&1 || true"
)

os.system(
    "nohup streamlit run /content/app.py "
    "--server.address 0.0.0.0 "
    "--server.port 8501 "
    "> /content/streamlit.log 2>&1 &"
)

print("✅ Streamlit started.")
print()
print("Check the Streamlit log with:")
print("!cat /content/streamlit.log")