# /Users/nabilsalehiyan/MilkCrate/scripts/benchmark_models_plus.py
import os, json, argparse, numpy as np, pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Optional libs (skip gracefully if missing)
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

def load_data(csv, label):
    df = pd.read_csv(csv)
    df.columns = [str(c).strip().lower() for c in df.columns]
    lbl = (label or "class").strip().lower()
    assert lbl in df.columns, f"Label '{lbl}' not in CSV"
    drop = [c for c in (lbl, "unnamed: 0", "id") if c in df.columns]
    X = df.drop(columns=drop, errors="ignore").select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
    y = df[lbl].astype(str).fillna("unknown")
    return X, y

def fit_eval(name, pipe, Xtr, Xte, ytr, yte, want_proba=True):
    pipe.fit(Xtr, ytr)
    yp = pipe.predict(Xte)
    acc = accuracy_score(yte, yp)
    f1m = f1_score(yte, yp, average="macro")
    row = {"model": name, "acc": acc, "macro_f1": f1m}
    if want_proba and hasattr(pipe, "predict_proba"):
        p = pipe.predict_proba(Xte)
        for k in (1, 3, 5):
            row[f"top{k}"] = top_k_accuracy_score(yte, p, k=k)
    return row, pipe

def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--csv", default=repo/"data/beatport_201611/beatsdataset_full.csv")
    ap.add_argument("--label", default="class")
    ap.add_argument("--outdir", default=repo/"artifacts")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Data
    X, y_str = load_data(args.csv, args.label)
    le = LabelEncoder(); y = le.fit_transform(y_str)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    imput = ("imp", SimpleImputer(strategy="median"))

    candidates = []

    # Strong baselines you already liked
    candidates.append(("rf",  Pipeline([imput, ("rf",  RandomForestClassifier(n_estimators=900, n_jobs=-1, random_state=42))]), True))
    candidates.append(("hgb", Pipeline([imput, ("hgb", HistGradientBoostingClassifier(max_iter=800, learning_rate=0.10, random_state=42))]), True))
    candidates.append(("extratrees", Pipeline([imput, ("et",  ExtraTreesClassifier(n_estimators=900, n_jobs=-1, random_state=42))]), True))

    # XGBoost
    if XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=900, learning_rate=0.08, max_depth=7,
            subsample=0.9, colsample_bytree=0.9,
            tree_method="hist", eval_metric="mlogloss", random_state=42, n_jobs=-1
        )
        candidates.append(("xgb", Pipeline([imput, ("xgb", xgb)]), True))

    # LightGBM
    if LGBMClassifier is not None:
        lgbm = LGBMClassifier(
            n_estimators=1200, learning_rate=0.05, num_leaves=63,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=0.0, random_state=42, n_jobs=-1
        )
        candidates.append(("lgbm", Pipeline([imput, ("lgbm", lgbm)]), True))

    # CatBoost
    if CatBoostClassifier is not None:
        cb = CatBoostClassifier(
            iterations=1200, learning_rate=0.08, depth=8,
            loss_function="MultiClass", verbose=False, random_seed=42
        )
        candidates.append(("catboost", Pipeline([imput, ("cb", cb)]), True))

    # SVM (RBF) — slower but sometimes strong; needs scaling; enable probabilities
    candidates.append(("svm_rbf", Pipeline([
        imput, ("sc", StandardScaler(with_mean=False)),
        ("svc", SVC(C=3.0, gamma="scale", kernel="rbf", probability=True, cache_size=1024, random_state=42))
    ]), True))

    # MLP — scale; early stopping
    candidates.append(("mlp", Pipeline([
        imput, ("sc", StandardScaler(with_mean=False)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(256,128), activation="relu",
                              alpha=1e-4, learning_rate_init=1e-3,
                              max_iter=400, early_stopping=True, random_state=42))
    ]), True))

    rows, fitted = [], {}
    for name, pipe, want_proba in candidates:
        try:
            res, fitted_pipe = fit_eval(name, pipe, Xtr, Xte, ytr, yte, want_proba=want_proba)
            rows.append(res)
            fitted[name] = fitted_pipe
            print(name, "→", {k: round(v,4) for k,v in res.items() if k!="model"})
        except Exception as e:
            print(f"[SKIP] {name}: {e}")

    lb = pd.DataFrame(rows).sort_values(["macro_f1","acc"], ascending=False)
    lb.to_csv(os.path.join(args.outdir, "benchmark_leaderboard_plus.csv"), index=False)
    print("\nLeaderboard saved → artifacts/benchmark_leaderboard_plus.csv")
    print(lb)

    # Soft-vote ensemble of best available probabilistic models
    ensemble = []
    for key in ("rf","hgb","lgbm","xgb","catboost","extratrees"):
        if key in fitted and hasattr(fitted[key], "predict_proba"):
            ensemble.append((key, fitted[key]))
    if len(ensemble) >= 2:
        ens = VotingClassifier(estimators=ensemble, voting="soft", weights=[1]*len(ensemble), n_jobs=None)
        ens.fit(Xtr, ytr)
        yp = ens.predict(Xte)
        acc = accuracy_score(yte, yp); f1m = f1_score(yte, yp, average="macro")
        row = {"model":"ensemble_soft", "acc":acc, "macro_f1":f1m}
        if hasattr(ens, "predict_proba"):
            p = ens.predict_proba(Xte)
            for k in (1,3,5):
                row[f"top{k}"] = top_k_accuracy_score(yte, p, k=k)
        lb = pd.concat([lb, pd.DataFrame([row])], ignore_index=True)
        lb.sort_values(["macro_f1","acc"], ascending=False, inplace=True)
        lb.to_csv(os.path.join(args.outdir, "benchmark_leaderboard_plus.csv"), index=False)
        print("Ensemble →", {k: round(v,4) for k,v in row.items() if k!="model"})
        fitted["ensemble_soft"] = ens

    # Save best
    best = lb.iloc[0]["model"]
    best_path = os.path.join(args.outdir, f"model_{best}.joblib")
    dump(fitted[best], best_path)
    # Save encoder and feature order (for inference)
    dump(le, os.path.join(args.outdir, "beatport201611_label_encoder.joblib"))
    with open(os.path.join(args.outdir, "beatport201611_feature_columns.json"), "w") as f:
        json.dump(list(X.columns), f)
    print(f"\nSaved best model → {best_path}")

if __name__ == "__main__":
    main()
