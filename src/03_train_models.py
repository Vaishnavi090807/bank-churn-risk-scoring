import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from utils import ensure_dir, save_pickle, save_json, load_pickle
from metrics import evaluate_at_threshold, find_best_threshold_min_fp

# -----------------------------
# CONFIG (Exact paths)
# -----------------------------
DATA_PATH = "../data/European_Bank.csv"
ARTIFACT_DIR = "../artifacts"

PREPROCESSOR_PATH = f"{ARTIFACT_DIR}/preprocessor.pkl"   # from Step 2
FEATURE_COLS_PATH = f"{ARTIFACT_DIR}/feature_columns.json"

BEST_MODEL_PATH = f"{ARTIFACT_DIR}/best_model.pkl"
METRICS_PATH = f"{ARTIFACT_DIR}/metrics.json"
THRESHOLD_REPORT_PATH = f"{ARTIFACT_DIR}/threshold_report.csv"

TARGET_COL = "Exited"
DROP_COLS = ["CustomerId", "Surname"]  # non-informative


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    df["Balance_to_Salary"] = df["Balance"] / (df["EstimatedSalary"] + eps)
    df["Product_Density"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["Engagement_Product_Interaction"] = df["IsActiveMember"] * df["NumOfProducts"]
    return df


def main():
    # 1) Load + clean
    df = pd.read_csv(DATA_PATH)

    # missing values safe handling
    if df.isnull().sum().sum() > 0:
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    # drop non-informative cols
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=c)

    # feature engineering (same as step 2)
    df = add_engineered_features(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # 2) Train-test split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 3) Load preprocessor (from Step 2)
    preprocessor = load_pickle(PREPROCESSOR_PATH)

    # 4) Define models (guide)
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight=None),
        "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=None),
        "RandomForest": RandomForestClassifier(
            random_state=42, n_estimators=300, max_depth=None, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }

    # 5) Train + Evaluate (base threshold=0.50) + ROC-AUC
    results = {}
    best_auc = -1
    best_name = None
    best_pipeline = None
    best_proba = None

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        # Evaluate at standard 0.50 threshold
        base_metrics = evaluate_at_threshold(y_test.values, y_proba, threshold=0.50)

        results[name] = {
            "base_threshold_metrics": base_metrics
        }

        # Choose best model by ROC-AUC (good general ranking metric)
        auc = base_metrics["roc_auc"]
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_pipeline = pipe
            best_proba = y_proba

        print(f"\n{name} @0.50 -> "
              f"AUC={auc:.4f}, "
              f"Prec={base_metrics['precision']:.4f}, "
              f"Recall={base_metrics['recall']:.4f}, "
              f"F1={base_metrics['f1']:.4f}")

    # 6) Threshold tuning to reduce false positives (secondary objective)
    #    Keep recall >= 0.60 (changeable), pick min FP.
    best_thresh, report = find_best_threshold_min_fp(
        y_true=y_test.values,
        y_proba=best_proba,
        min_recall=0.60
    )

    results["BEST_MODEL"] = {
        "name": best_name,
        "roc_auc": float(best_auc),
        "chosen_threshold_policy": "min false positives with recall constraint",
        "tuned_threshold_result": best_thresh
    }

    # save threshold report CSV
    ensure_dir(ARTIFACT_DIR)
    pd.DataFrame(report).to_csv(THRESHOLD_REPORT_PATH, index=False)

    # 7) Save best model pipeline + metrics
    save_pickle(best_pipeline, BEST_MODEL_PATH)
    save_json(results, METRICS_PATH)

    print("\nTraining complete.")
    print("Best model:", best_name)
    print("Saved best model:", BEST_MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("Saved threshold report:", THRESHOLD_REPORT_PATH)
    print("\nTuned threshold picked to reduce FPs:")
    print(best_thresh)


if __name__ == "__main__":
    main()