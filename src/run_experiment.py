"""Comparative text classification study on IMDb movie reviews.

Run:
  python src/run_experiment.py --csv dataset/IMDB Dataset.csv

The script creates metrics, plots, confusion matrices, and a what-if SVM comparison.
It does not include fabricated results; all outputs are computed from the supplied CSV.
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_data(csv_path: Path, samples_per_class: int, random_state: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"review", "sentiment"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")
    df = df[["review", "sentiment"]].dropna().copy()
    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    df = df[df["sentiment"].isin(["positive", "negative"])]
    counts = df["sentiment"].value_counts()
    if any(counts.get(label, 0) < samples_per_class for label in ["positive", "negative"]):
        raise ValueError(
            f"Need at least {samples_per_class} records per class; found {counts.to_dict()}"
        )
    df = (
        df.groupby("sentiment", group_keys=False)
          .sample(n=samples_per_class, random_state=random_state)
          .sample(frac=1, random_state=random_state)
          .reset_index(drop=True)
    )
    df["review"] = df["review"].map(clean_text)
    return df


def metric_row(name, y_true, y_pred):
    return {
        "Algorithm": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label="positive", zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label="positive", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, pos_label="positive", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(ROOT / "dataset" / "IMDB Dataset.csv"))
    parser.add_argument("--samples-per-class", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\nDownload the IMDb CSV and place it in dataset/."
        )

    df = load_data(csv_path, args.samples_per_class, args.random_state)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.20,
        random_state=args.random_state, stratify=df["sentiment"]
    )

    # Fit TF-IDF only on training text to avoid test-set leakage.
    vectorizer = TfidfVectorizer(
        max_features=10000, stop_words="english", ngram_range=(1, 2)
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=args.random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"),
        "SVM": LinearSVC(C=1.0, random_state=args.random_state),
        "Naive Bayes": MultinomialNB(),
    }

    rows, predictions = [], {}
    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        rows.append(metric_row(name, y_test, pred))
        (RESULTS / f"{name.lower().replace(' ', '_')}_classification_report.txt").write_text(
            classification_report(y_test, pred, zero_division=0), encoding="utf-8"
        )

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / "algorithm_comparison.csv", index=False)

    ax = results.set_index("Algorithm")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
        kind="bar", figsize=(11, 6), ylim=(0, 1), width=0.8
    )
    ax.set_title("Performance Comparison of Four Text Classifiers")
    ax.set_ylabel("Score")
    ax.set_xlabel("Algorithm")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / "performance_comparison.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (name, pred) in zip(axes.flatten(), predictions.items()):
        cm = confusion_matrix(y_test, pred, labels=["negative", "positive"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"]
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(RESULTS / "confusion_matrices.png", dpi=180)
    plt.close()

    # What-if experiment: change only SVM C; keep data and features unchanged.
    what_if_rows = []
    for c_value in [1.0, 0.1]:
        model = LinearSVC(C=c_value, random_state=args.random_state)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        row = metric_row(f"SVM (C={c_value})", y_test, pred)
        row["C"] = c_value
        what_if_rows.append(row)
    what_if = pd.DataFrame(what_if_rows)
    what_if.to_csv(RESULTS / "svm_what_if_comparison.csv", index=False)

    best = results.sort_values(["F1-score", "Accuracy"], ascending=False).iloc[0]
    summary = [
        "EXPERIMENT SUMMARY — GENERATED FROM YOUR RUN",
        f"Dataset sample: {len(df)} reviews ({args.samples_per_class} per class)",
        f"Train records: {len(X_train_text)}",
        f"Test records: {len(X_test_text)}",
        "Positive class used for Precision/Recall/F1: positive",
        f"Highest F1-score in this run: {best['Algorithm']} ({best['F1-score']:.4f})",
        "",
        "Full metrics are in algorithm_comparison.csv.",
        "Do not claim this model is universally best; the conclusion applies to this dataset and setup."
    ]
    (RESULTS / "run_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("\n=== ACTUAL RESULTS FROM THIS RUN ===")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== SVM WHAT-IF RESULTS ===")
    print(what_if.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nBest F1-score in this run: {best['Algorithm']} ({best['F1-score']:.4f})")
    print(f"Files saved in: {RESULTS}")


if __name__ == "__main__":
    main()
