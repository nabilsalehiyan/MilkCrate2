import os
from joblib import load

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_le = os.path.join(repo_root, "artifacts", "beatport201611_label_encoder.joblib")

    le_path = default_le
    if not os.path.exists(le_path):
        raise SystemExit(
            f"Label encoder not found at:\n  {le_path}\n"
            f"Check your artifacts folder."
        )

    le = load(le_path)
    classes = list(getattr(le, "classes_", []))
    print(f"Total classes: {len(classes)}")
    for i, c in enumerate(classes):
        print(f"{i:2d} -> {c}")

if __name__ == "__main__":
    main()
