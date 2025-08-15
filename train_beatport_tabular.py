# /Users/nabilsalehiyan/MilkCrate/scripts/train_beatport_tabular.py
import os, json, argparse
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import dump

LABEL_CANDIDATES = ("class", "subgenre", "sub_genre", "genre", "main_genre", "style")

def parse_args():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    p = argparse.ArgumentParser("Train Beatport tabular classifier")
    p.add_argument("--csv", default=os.path.join(repo, "data", "beatport_201611", "beatsdataset_full.csv"),
                   help="Path to training CSV")
    p.add_argument("--label", default=None,
                   help="Label column name (default: auto-detect among class/genre/subgenre/style)")
    p.add_argument("--outdir", default=os.path.join(repo, "artifacts"),
                   help="Artifacts output dir")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--n_estimators", type=int, default=600)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def pick_label(df: pd.DataFrame, requested: str | None):
    if requested:
        r = requested.strip().lower()
        if r in df.columns: return r
    for c in LABEL_CANDIDATES:
        if c in df.columns: return c
    return None

def main():
    a = parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    csv = os.path.abspath(a.csv)
    if not os.path.exists(csv):
        raise SystemExit(f"CSV not found: {csv}")
    if "songs" in os.path.basename(csv).lower():
        raise SystemExit(f"Wrong CSV (lookup table): {csv} — use beatsdataset_full.csv")

    print(f"[INFO] Loading: {csv}")
    df_raw = pd.read_csv(csv)
    print(f"[INFO] Raw columns: {len(df_raw.columns)}")
    print(f"[INFO] First 10 cols: {list(df_raw.columns)[:10]}")
    print(f"[INFO] Last 10 cols:  {list(df_raw.columns)[-10:]}")

    df = normalize_headers(df_raw)
    label = pick_label(df, a.label)
    if not label:
        raise SystemExit(
            "Could not find a label column.\n"
            f"Tried requested='{a.label}' and candidates={LABEL_CANDIDATES}\n"
            f"Available (tail): {list(df.columns)[-25:]}"
        )
    print(f"[INFO] Using label column: '{label}'")

    # Build features
    drop = [c for c in (label, "unnamed: 0", "id") if c in df.columns]
    if drop: print(f"[INFO] Dropping non-feature cols: {drop}")
    X = df.drop(columns=drop, errors="ignore").select_dtypes(include=["number"])
    if X.shape[1] == 0:
        raise SystemExit("No numeric feature columns found.")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Labels
    y_str = df[label].astype(str).fillna("unknown")
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    print(f"[INFO] Classes ({len(le.classes_)}): {sorted(le.classes_)[:10]}{' ...' if len(le.classes_)>10 else ''}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=a.test_size, random_state=a.random_state,
        stratify=y if len(set(y)) > 1 else None
    )
    print(f"[INFO] Train: {Xtr.shape} | Test: {Xte.shape}")

    # Model
    clf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=a.n_estimators, n_jobs=-1, random_state=a.random_state
        ))
    ])
    clf.fit(Xtr, ytr)

    yp = clf.predict(Xte)
    acc = accuracy_score(yte, yp)
    f1m = f1_score(yte, yp, average="macro")
    report = classification_report(yte, yp, target_names=le.classes_, digits=3)
    print("\n=== Classification Report ===\n" + report)
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1m:.4f}")

    # Save artifacts
    dump(clf, os.path.join(a.outdir, "beatport201611_rf.joblib"))
    dump(le,  os.path.join(a.outdir, "beatport201611_label_encoder.joblib"))
    with open(os.path.join(a.outdir, "beatport201611_feature_columns.json"), "w") as f:
        json.dump(list(X.columns), f)
    with open(os.path.join(a.outdir, "beatport201611_class_names.json"), "w") as f:
        json.dump(list(le.classes_), f, indent=2)
    with open(os.path.join(a.outdir, "beatport201611_metrics.json"), "w") as f:
        json.dump({"accuracy": acc, "macro_f1": f1m}, f, indent=2)
    with open(os.path.join(a.outdir, "beatport201611_class_report.txt"), "w") as f:
        f.write(report)

    print("\n[INFO] Saved artifacts to:", a.outdir)

if __name__ == "__main__":
    main()
