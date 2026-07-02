"""
Classical ML baseline: TF-IDF + Logistic Regression resume category classifier.
Serves as a sanity-check / comparison point against the embedding-based matcher.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline


def train_baseline(df, text_col="clean_text", label_col="Category", test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df[label_col], test_size=test_size, random_state=random_state, stratify=df[label_col]
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    return pipe, {"accuracy": acc, "report": report, "y_test": y_test, "y_pred": y_pred}


if __name__ == "__main__":
    from src.data_prep import load_and_clean
    df = load_and_clean("/mnt/user-data/uploads/resumes_dataset.jsonl")
    pipe, results = train_baseline(df)
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(results["report"][:2000])
