
import os
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Parking Occupancy — Hardcoded Pickle + Preprocessing", layout="wide")
st.title("🅿️ Parking Slot Occupancy — Hardcoded Pickle + Preprocessing")

# =========================================================
# 1) HARD-CODE THESE PATHS
#    - HARDCODED_MODEL_PATH: required (your classifier/regressor pickle)
#    - HARDCODED_ENCODERS_PATH: optional (a joblib.pkl containing LabelEncoders or dicts)
# =========================================================
HARDCODED_MODEL_PATH = "model.pkl"     # <-- EDIT THIS
HARDCODED_ENCODERS_PATH = None                        # e.g., "models/encoders.pkl" if you saved LabelEncoders

# If your model expects features in a specific order, set it here:
FEATURE_ORDER = ["Day_of_Week", "Hour", "Weather", "Nearby_Events", "Slot_ID"]

# ---------------------------------------------------------
# Default categorical mappings (used if no encoders.pkl provided)
# Adjust to match how you encoded during training.
# ---------------------------------------------------------
DEFAULT_MAPS = {
    "Day_of_Week": {"Mon":0, "Tue":1, "Wed":2, "Thu":3, "Fri":4, "Sat":5, "Sun":6},
    "Weather": {"Clear":0, "Cloudy":1, "Rain":2, "Storm":3},
    "Nearby_Events": {"None":0, "Market":1, "Stadium":2, "Concert":3},
}

# Keep a dynamic mapping for Slot_ID strings if your training used numeric IDs
if "slot_id_map" not in st.session_state:
    st.session_state.slot_id_map = {}

def slot_to_number(x):
    # If already numeric, return as int
    try:
        return int(x)
    except Exception:
        pass
    # Try to extract a number from strings like "S12" -> 12
    if isinstance(x, str):
        m = re.search(r"(\d+)", x)
        if m:
            return int(m.group(1))
        # fallback: stable map per session
        if x not in st.session_state.slot_id_map:
            st.session_state.slot_id_map[x] = len(st.session_state.slot_id_map) + 1
        return st.session_state.slot_id_map[x]
    # Unknown type -> 0
    return 0

# ---------------------------------------------------------
# Optional: load encoders (dict of sklearn LabelEncoders or mappings)
# ---------------------------------------------------------
encoders = None
if HARDCODED_ENCODERS_PATH:
    try:
        if os.path.exists(HARDCODED_ENCODERS_PATH):
            encoders = joblib.load(HARDCODED_ENCODERS_PATH)
    except Exception as e:
        st.warning(f"Could not load encoders from {HARDCODED_ENCODERS_PATH}: {e}")

def apply_encoders_or_maps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Day_of_Week
    if "Day_of_Week" in out.columns:
        col = out["Day_of_Week"].astype(str)
        if encoders and "Day_of_Week" in encoders:
            out["Day_of_Week"] = encoders["Day_of_Week"].transform(col)
        else:
            out["Day_of_Week"] = col.map(DEFAULT_MAPS["Day_of_Week"]).astype("Int64")
    # Weather
    if "Weather" in out.columns:
        col = out["Weather"].astype(str)
        if encoders and "Weather" in encoders:
            out["Weather"] = encoders["Weather"].transform(col)
        else:
            out["Weather"] = col.map(DEFAULT_MAPS["Weather"]).astype("Int64")
    # Nearby_Events
    if "Nearby_Events" in out.columns:
        col = out["Nearby_Events"].astype(str)
        if encoders and "Nearby_Events" in encoders:
            out["Nearby_Events"] = encoders["Nearby_Events"].transform(col)
        else:
            out["Nearby_Events"] = col.map(DEFAULT_MAPS["Nearby_Events"]).astype("Int64")
    # Hour
    if "Hour" in out.columns:
        out["Hour"] = pd.to_numeric(out["Hour"], errors="coerce").fillna(0).astype(int)
    # Slot_ID
    if "Slot_ID" in out.columns:
        out["Slot_ID"] = out["Slot_ID"].apply(slot_to_number).astype(int)

    # Handle unseen categories that map to <NA>
    for c in ["Day_of_Week", "Weather", "Nearby_Events"]:
        if c in out.columns:
            out[c] = out[c].fillna(-1).astype(int)  # unseen -> -1

    # Reorder columns
    out = out[[c for c in FEATURE_ORDER if c in out.columns]]
    return out

# ---------------------------------------------------------
# Load the model at startup
# ---------------------------------------------------------
pipe = None
load_error = None
try:
    if not os.path.exists(HARDCODED_MODEL_PATH):
        load_error = f"File not found: {HARDCODED_MODEL_PATH}"
    else:
        pipe = joblib.load(HARDCODED_MODEL_PATH)
except Exception as e:
    load_error = f"Failed to load model: {e}"

if pipe is not None:
    st.success(f"✅ Loaded model: {HARDCODED_MODEL_PATH} ({type(pipe)})")
else:
    st.error(load_error or "Unknown error loading model.")
    st.stop()

st.caption("If your training used LabelEncoders, set HARDCODED_ENCODERS_PATH to load them. "
           "Otherwise the app uses default numeric mappings above.")

# -----------------------
# Single prediction
# -----------------------
st.subheader("🔮 Single Prediction")
c1, c2, c3 = st.columns(3)
with c1:
    Day_of_Week = st.selectbox("Day_of_Week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    Weather = st.selectbox("Weather", ["Clear","Cloudy","Rain","Storm"])
with c2:
    Nearby_Events = st.selectbox("Nearby_Events", ["None","Market","Stadium","Concert"])
    Hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=8, step=1)
with c3:
    Slot_ID = st.text_input("Slot_ID", value="S1")

if st.button("Predict single"):
    row = pd.DataFrame([{
        "Day_of_Week": Day_of_Week,
        "Hour": int(Hour),
        "Weather": Weather,
        "Nearby_Events": Nearby_Events,
        "Slot_ID": Slot_ID
    }])
    X = apply_encoders_or_maps(row)
    try:
        pred = pipe.predict(X)[0]
        proba = None
        if hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X)[0, 1]
            except Exception:
                proba = None
        st.success(f"Prediction — Slot_Occupied = **{int(pred)}**")
        if proba is not None:
            st.write(f"Probability(occupied=1): **{proba:.3f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()

# -----------------------
# Batch predictions
# -----------------------
st.subheader("📦 Batch Predictions from CSV")
st.caption("Upload a CSV with columns: Day_of_Week, Hour, Weather, Nearby_Events, Slot_ID")
up = st.file_uploader("Upload features CSV", type=["csv"], key="csv")
if up is not None:
    try:
        df = pd.read_csv(up)
        st.write("Preview:")
        st.dataframe(df.head(), use_container_width=True)

        required = [c for c in FEATURE_ORDER if c in df.columns]
        missing = [c for c in FEATURE_ORDER if c not in df.columns]
        if missing:
            st.warning(f"Missing columns in upload: {missing}. Proceeding with available ones: {required}")
        Xb = apply_encoders_or_maps(df[required])
        if st.button("Predict batch"):
            preds = pipe.predict(Xb)
            out = df.copy()
            out["Slot_Occupied_Pred"] = preds
            if hasattr(pipe, "predict_proba"):
                try:
                    proba = pipe.predict_proba(Xb)[:, 1]
                    out["Prob_Occupied"] = proba
                except Exception:
                    pass
            st.success("✅ Predictions ready")
            st.dataframe(out.head(), use_container_width=True)

            csv_data = out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download predictions CSV",
                               data=csv_data,
                               file_name="predictions.csv",
                               mime="text/csv")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
