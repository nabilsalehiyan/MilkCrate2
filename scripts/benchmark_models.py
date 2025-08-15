# /Users/nabilsalehiyan/MilkCrate/scripts/benchmark_models.py
import os, json, argparse, numpy as np, pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint, uniform

def load_data(csv, label):
    df = pd.read_csv(csv)
    df.columns = [str(c).strip().lower() for c in df.columns]
    lbl = (label or "class").strip().lower()
    assert lbl in df.columns, f"Label '{lbl}' not in CSV"
    drop = [c for c in (lbl, "unnamed: 0", "id") if c in df.columns]
    X = df.drop(columns=drop, errors="ignore").select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
    y = df[lbl].astype(str).fillna("unknown")
    return X, y

def fit_eval(name, pipe, Xtr, Xte, ytr, yte, proba=True):
    pipe.fit(Xtr, ytr)
    yp = pipe.predict(Xte)
    acc = accuracy_score(yte, yp)
    f1m = f1_score(yte, yp, average="macro")
    scores = {"model":name, "acc":acc, "macro_f1":f1m}
    if proba and hasattr(pipe, "predict_proba"):
        p = pipe.predict_proba(Xte)
        for k in (1,3,5):
            scores[f"top{k}"] = top_k_accuracy_score(yte, p, k=k)
    return scores, pipe

def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--csv", default=repo/"data/beatport_201611/beatsdataset_full.csv")
    ap.add_argument("--label", default="class")
    ap.add_argument("--outdir", default=repo/"artifacts")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    X, y_str = load_data(args.csv, args.label)
    le = LabelEncoder(); y = le.fit_transform(y_str)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    imput = ("imp", SimpleImputer(strategy="median"))

    # Candidates
    models = []

    # 1) RandomForest (quick strong baseline)
    rf = Pipeline([imput, ("rf", RandomForestClassifier(n_estimators=800, n_jobs=-1, random_state=42))])
    models.append(("rf", rf, True))

    # 2) ExtraTrees
    et = Pipeline([imput, ("et", ExtraTreesClassifier(n_estimators=800, n_jobs=-1, random_state=42))])
    models.append(("extratrees", et, True))

    # 3) HistGradientBoosting (a good booster)
    hgb = Pipeline([imput, ("hgb", HistGradientBoostingClassifier(random_state=42, max_iter=600, learning_rate=0.12))])
    models.append(("hgb", hgb, True))

    # 4) kNN (sometimes good on dense features)
    knn = Pipeline([imput, ("sc", StandardScaler(with_mean=False)), ("knn", KNeighborsClassifier(n_neighbors=25))])
    models.append(("knn", knn, False))  # no proba by default in this pipeline

    # 5) RF (tuned, light random search)
    rf_rs = Pipeline([imput, ("rf", RandomForestClassifier(n_jobs=-1, random_state=42))])
    rf_search = RandomizedSearchCV(
        rf_rs,
        {
            "rf__n_estimators": randint(500, 1200),
            "rf__max_depth": randint(10, 40),
            "rf__min_samples_split": randint(2, 20),
            "rf__min_samples_leaf": randint(1, 8),
        },
        n_iter=20, cv=3, scoring="f1_macro", n_jobs=-1, random_state=42, verbose=0
    )
    models.append(("rf_tuned", rf_search, True))

    rows, fitted = [], {}
    for name, pipe, want_proba in models:
        s, m = fit_eval(name, pipe, Xtr, Xte, ytr, yte, proba=want_proba)
        rows.append(s); fitted[name] = m.best_estimator_ if hasattr(m, "best_estimator_") else m
        print(name, "→", {k: round(v,4) for k,v in s.items() if k!="model"})

    lb = pd.DataFrame(rows).sort_values(["macro_f1","acc"], ascending=False)
    lb.to_csv(os.path.join(args.outdir, "benchmark_leaderboard.csv"), index=False)
    print("\nLeaderboard saved → artifacts/benchmark_leaderboard.csv")
    print(lb.head())

    # Save the best model and a copy of the encoder/feats
    best = lb.iloc[0]["model"]
    best_path = os.path.join(args.outdir, f"model_{best}.joblib")
    dump(fitted[best], best_path)
    dump(le, os.path.join(args.outdir, "beatport201611_label_encoder.joblib"))
    with open(os.path.join(args.outdir, "beatport201611_feature_columns.json"), "w") as f:
        json.dump(list(X.columns), f)
    print(f"Saved best model → {best_path}")

if __name__ == "__main__":
    main()
