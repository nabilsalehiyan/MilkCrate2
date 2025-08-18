
import os, io, json, joblib, numpy as np, pandas as pd, streamlit as st

DEFAULT_MODEL_PATH = "artifacts/beatport201611_hgb.joblib"       # 46MB model in repo
DEFAULT_ENCODER_PATH = "artifacts/label_encoder.joblib"          # 23-class encoder
DEFAULT_CSV_PATH = "data/beatport_features.csv"                  # only used if we rebuild encoder
TARGET_COL_DEFAULT = "genre"

st.set_page_config(page_title="MilkCrate Genre Classifier", layout="wide")

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model not found at {path}")
        st.stop()
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_encoder(enc_path: str):
    if not os.path.exists(enc_path):
        st.error(f"Encoder not found at {enc_path}.")
        st.stop()
    return joblib.load(enc_path)

def align_columns_to_model(X: pd.DataFrame, model):
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        missing = [c for c in names if c not in X.columns]
        if missing:
            st.warning(f"Missing {len(missing)} expected columns. First few: {missing[:10]}")
        X = X.reindex(columns=names)
    return X

def get_display_names(model, encoder):
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None, None
    classes = np.array(classes)
    if np.issubdtype(classes.dtype, np.number):
        names = encoder.inverse_transform(classes.astype(int))
        return classes, names
    return classes, classes

def predict_one(model, encoder, features_df: pd.DataFrame, top_k=5):
    X = features_df.select_dtypes(include=[np.number])
    X = align_columns_to_model(X, model)
    y = model.predict(X)
    y_labels = encoder.inverse_transform(y.astype(int)) if np.issubdtype(y.dtype, np.number) else y
    out = {"pred_idx": y, "pred_label": y_labels}
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0]
        classes, names = get_display_names(model, encoder)
        order = np.argsort(prob)[::-1][: min(top_k, len(prob))]
        out["topk"] = pd.DataFrame({
            "rank": np.arange(1, len(order)+1),
            "label": [names[i] for i in order],
            "class_code": [int(classes[i]) if np.issubdtype(classes.dtype, np.integer) else classes[i] for i in order],
            "probability": [float(prob[i]) for i in order],
        })
    return out

def read_uploaded_json_or_row(uploaded_bytes: bytes):
    text = uploaded_bytes.decode("utf-8", errors="ignore")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict): return pd.DataFrame([obj])
        if isinstance(obj, list) and obj and isinstance(obj[0], dict): return pd.DataFrame(obj[:1])
    except Exception:
        pass
    try:
        df = pd.read_csv(io.StringIO(text))
        return df.head(1)
    except Exception as e:
        st.error("Couldn't parse as JSON or CSV."); st.caption(str(e)); return None

st.title("🎛️ MilkCrate — Genre Classifier")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    encoder_path = st.text_input("Encoder path", value=DEFAULT_ENCODER_PATH)
    top_k = st.number_input("Top-K", min_value=1, max_value=10, value=5, step=1)

model = load_model(model_path)
encoder = load_encoder(encoder_path)

with st.expander("🔎 Debug: label map"):
    classes, names = get_display_names(model, encoder)
    if classes is not None:
        st.dataframe(pd.DataFrame({"class_code": classes, "label": names}), use_container_width=True, hide_index=True)

tabs = st.tabs(["Single item", "Batch CSV"])

with tabs[0]:
    st.subheader("Single item prediction")
    up = st.file_uploader("Upload features (JSON dict or 1-row CSV)", type=["json","csv"], key="single")
    if up is not None:
        df = read_uploaded_json_or_row(up.getvalue())
        if df is not None:
            st.write("Parsed input:"); st.dataframe(df, use_container_width=True)
            try:
                res = predict_one(model, encoder, df, top_k=top_k)
                st.success(f"Predicted genre: **{res['pred_label'][0]}**")
                if "topk" in res: st.dataframe(res["topk"], use_container_width=True, hide_index=True)
            except Exception as e:
                st.error("Prediction failed."); st.exception(e)

with tabs[1]:
    st.subheader("Batch predictions from CSV")
    upcsv = st.file_uploader("Upload features CSV", type=["csv"], key="batch")
    if upcsv is not None:
        try: df = pd.read_csv(upcsv)
        except Exception as e: st.error("Couldn't read CSV."); st.caption(str(e)); df=None
        if df is not None:
            st.dataframe(df.head(10), use_container_width=True)
            X = align_columns_to_model(df.select_dtypes(include=[np.number]), model)
            try:
                y_pred = model.predict(X)
                y_label = encoder.inverse_transform(y_pred.astype(int)) if np.issubdtype(y_pred.dtype, np.number) else y_pred
                out = pd.DataFrame({"pred_idx": y_pred, "pred_label": y_label})
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                    classes, names = get_display_names(model, encoder)
                    top_idx = np.argsort(proba, axis=1)[:, ::-1][:, :top_k]
                    out["top_labels"] = [[names[i] for i in row] for row in top_idx]
                    out["top_probs"] = [[float(proba[r,i]) for i in row] for r,row in enumerate(top_idx)]
                st.dataframe(out.head(100), use_container_width=True)
                if pd.Series(out["pred_label"]).nunique(dropna=False) == 1:
                    st.warning("All predictions are the same class in this preview — check class balance or train/inference feature mismatch.")
            except Exception as e:
                st.error("Prediction failed."); st.exception(e)

st.caption("MilkCrate • human-readable labels via LabelEncoder")
