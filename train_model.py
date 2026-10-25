"""Train and evaluate the customer support ticket classifier."""

from pathlib import Path
from difflib import SequenceMatcher

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "customer_support_tickets.csv"
CATEGORIES = ["Billing", "Technical Support", "Account Access", "Order Issues"]
RANDOM_STATE = 42


def preprocess_text(text: str) -> str:
    """Apply the same conservative cleaning during training and prediction."""
    return " ".join(str(text).lower().strip().split())


def remove_near_duplicates(texts: pd.Series, threshold: float = 0.94) -> list[int]:
    """Return duplicate row indexes, including highly similar later rows."""
    duplicate_indexes: list[int] = []
    normalized = texts.map(preprocess_text)
    seen: list[tuple[int, str]] = []
    for index, text in normalized.items():
        if any(
            SequenceMatcher(None, text, previous).ratio() >= threshold
            for _, previous in seen
        ):
            duplicate_indexes.append(index)
        else:
            seen.append((index, text))
    return duplicate_indexes


def inspect_and_clean_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)
    required_columns = {"ticket_text", "category"}
    if not required_columns.issubset(data.columns):
        raise ValueError("Dataset must contain ticket_text and category columns.")

    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print("\nSample records:")
    print(data.head())
    print("\nMissing values:")
    print(data[["ticket_text", "category"]].isna().sum())
    print(f"Exact duplicate rows: {data.duplicated().sum()}")
    print("\nCategory counts before cleaning:")
    print(data["category"].value_counts())

    data = data[["ticket_text", "category"]].dropna()
    data["ticket_text"] = data["ticket_text"].map(preprocess_text)
    data["category"] = data["category"].astype(str).str.strip()
    data = data[data["ticket_text"].str.len() > 0]
    data = data[data["category"].isin(CATEGORIES)].drop_duplicates(
        subset=["ticket_text"]
    )

    near_duplicates = remove_near_duplicates(data["ticket_text"])
    print(f"Near-duplicate rows identified for removal: {len(near_duplicates)}")
    if near_duplicates:
        data = data.drop(index=near_duplicates)

    if set(data["category"]) != set(CATEGORIES):
        raise ValueError("Every required category must remain after cleaning.")
    print("\nCategory counts after cleaning:")
    print(data["category"].value_counts())
    return data.reset_index(drop=True)


def save_confusion_matrix(
    matrix: np.ndarray, model_name: str, output_path: Path
) -> None:
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CATEGORIES,
        yticklabels=CATEGORIES,
    )
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted category")
    plt.ylabel("True category")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    data = inspect_and_clean_dataset()
    train_text, test_text, train_labels, test_labels = train_test_split(
        data["ticket_text"],
        data["category"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data["category"],
    )
    if set(train_labels) != set(CATEGORIES) or set(test_labels) != set(CATEGORIES):
        raise ValueError("Train and test sets must contain every category.")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_df=0.98
    )
    train_features = vectorizer.fit_transform(train_text)
    test_features = vectorizer.transform(test_text)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
    }
    results: dict[str, dict[str, float]] = {}
    trained_models = {}
    for name, model in models.items():
        model.fit(train_features, train_labels)
        predictions = model.predict(test_features)
        results[name] = {
            "macro_f1": f1_score(test_labels, predictions, average="macro"),
            "macro_precision": precision_score(
                test_labels, predictions, average="macro", zero_division=0
            ),
            "macro_recall": recall_score(
                test_labels, predictions, average="macro", zero_division=0
            ),
        }
        trained_models[name] = model
        print(f"\n{name} classification report:")
        print(
            classification_report(
                test_labels, predictions, labels=CATEGORIES, zero_division=0
            )
        )
        save_confusion_matrix(
            confusion_matrix(test_labels, predictions, labels=CATEGORIES),
            name,
            BASE_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png",
        )

    comparison = pd.DataFrame(results).T.sort_values("macro_f1", ascending=False)
    print("\nModel comparison:")
    print(comparison)
    best_name = comparison.index[0]
    joblib.dump(trained_models[best_name], BASE_DIR / "ticket_classifier.pkl")
    joblib.dump(vectorizer, BASE_DIR / "tfidf_vectorizer.pkl")
    print(f"\nSelected {best_name} using macro-F1.")
    print("Saved ticket_classifier.pkl and tfidf_vectorizer.pkl.")


if __name__ == "__main__":
    main()
