# /Users/nabilsalehiyan/MilkCrate/app.py
import os, sys, importlib.util, tempfile, pathlib
import pandas as pd
import streamlit as st

# Ensure repo root on path so local package imports work both locally and on Streamlit Cloud
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- Try normal imports; fallback to loading modules by path if needed ---
try:
    from milkcrate.inference_tabular import predict_dataframe, get_active_model_info
    from milkcrate.audio_features import features_dataframe_from_audio_paths
except ModuleNotFoundError:
    # inference_tabular fallback
    spec_inf = importlib.util.spec_from_file_location(
        "milkcrate.inference_tabular", os.path.join(ROOT, "milkcrate", "inference_tabular.py")
    )
    mod_inf = importlib.util.module_from_spec(spec_inf); spec_inf.loader.exec_module(mod_inf)
    predict_dataframe = mod_inf.predict_dataframe
    get_active_model_info = mod_inf.get_active_model_info
    # audio_features fallback
    spec_aud = importlib.util.spec_from_file_location(
        "milkcrate.audio_features", os.path.join(ROOT, "milkcrate", "audio_features.py")
    )
    mod_aud = importlib.util.module_from_spec(spec_aud); spec_aud.loader.exec_module(mod_aud)
    features_dataframe_from_audio_paths = mod_aud.features_dataframe_from_audio_paths

# ---------------- UI ----------------
st.set_page_config(page_title="MilkCrate", layout="centered")
st.title("MilkCrate – Audio Genre Classifier")

# Show which model/version is live (don’t crash UI if artifacts missing)
try:
    info = get_active_model_info()
    st.caption(f"Model: {info.get('model_version') or 'fallback'} • file: {info['model_file']}")
except Exception as e:
    st.warning(f"Model not loaded yet: {e}")

st.subheader("Upload audio files")
st.caption("Supported: WAV, MP3, OGG, FLAC, M4A. If MP3 decoding fails on your platform, convert to WAV and try again.")
uploads = st.file_uploader(
    "Drop one or more audio files",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
    accept_multiple_files=True,
)

def _extract_top1_conf(val):
    """Extract top-1 probability from a `topk=[(label, prob)]` cell, else None."""
    try:
        if isinstance(val, list) and val and isinstance(val[0], (list, tuple)):
            return float(val[0][1])
    except Exception:
        pass
    return None

if uploads:
    # Persist uploaded files to temp paths for librosa
    tmp_paths, shown_names = [], []
    for upl in uploads:
        suffix = "." + pathlib.Path(upl.name).suffix.lstrip(".").lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upl.read())
            tmp_paths.append(tmp.name)
        shown_names.append(pathlib.Path(upl.name).name)

    try:
        with st.spinner("Extracting features and predicting…"):
            feats_df = features_dataframe_from_audio_paths(tmp_paths)
            # Predict. We request topk=1 only so we can derive confidence, then drop the column.
            feats_only = feats_df.drop(columns=[c for c in ["source_path"] if c in feats_df.columns])
            preds = predict_dataframe(feats_only, include_topk=1)

        # Attach original filenames
        preds.insert(0, "file", shown_names)

        # Build top-1 view: label + optional confidence
        display = preds.copy()
        if "topk" in display.columns:
            display["confidence"] = display["topk"].apply(_extract_top1_conf)
            display.drop(columns=["topk"], inplace=True, errors="ignore")

        cols = ["file", "predicted_class"]
        if "confidence" in display.columns:
            cols.append("confidence")

        st.success("Done.")
        st.dataframe(display[cols], use_container_width=True)

        # CSV-friendly download
        csv_out = display[cols].copy()
        st.download_button(
            "Download predictions.csv",
            csv_out.to_csv(index=False),
            "predictions.csv",
            "text/csv",
        )
    except Exception as e:
        st.error(f"Failed to process audio: {e}")

st.markdown("---")
with st.expander("CSV mode (optional)"):
    csv = st.file_uploader("Upload features CSV", type=["csv"], accept_multiple_files=False, key="csvonly")
    if csv:
        try:
            df = pd.read_csv(csv)
            # Same approach: get top-1 confidence, then drop the internal topk column
            out = predict_dataframe(df, include_topk=1)
            disp = out.copy()
            if "topk" in disp.columns:
                disp["confidence"] = disp["topk"].apply(_extract_top1_conf)
                disp.drop(columns=["topk"], inplace=True, errors="ignore")
            cols = ["predicted_class", "confidence"] if "confidence" in disp.columns else ["predicted_class"]
            st.dataframe(disp[cols], use_container_width=True)
            st.download_button(
                "Download predictions.csv",
                disp[cols].to_csv(index=False),
                "predictions.csv",
                "text/csv",
            )
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")
