import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from utils import ensure_dir, save_pickle, save_json

# -----------------------------
# CONFIG (Exact paths)
# -----------------------------
DATA_PATH = "../data/European_Bank.csv"
ARTIFACT_DIR = "../artifacts"
SCALER_PATH = f"{ARTIFACT_DIR}/scaler.pkl"
FEATURE_COLS_PATH = f"{ARTIFACT_DIR}/feature_columns.json"

TARGET_COL = "Exited"
DROP_COLS = ["CustomerId", "Surname"]  # non-informative


def main():
    # 1) Load dataset
    df = pd.read_csv(DATA_PATH)

    print("Loaded:", df.shape)
    print("Columns:", df.columns.tolist())

    # 2) Missing values check
    missing = df.isnull().sum().sort_values(ascending=False)
    print("\nMissing values (top):")
    print(missing.head(10))

    # Optional: if any missing exists, simple handling
    # (usually this dataset has none)
    if df.isnull().sum().sum() > 0:
        # Fill numeric with median, categorical with mode
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
        print("\nMissing values handled.")

    # 3) Remove non-informative features
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=c)

    # 4) Split X and y
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Identify categorical + numeric columns
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if X[c].dtype != "object"]

    print("\nCategorical columns:", cat_cols)
    print("Numeric columns:", num_cols)

    # 5) Build preprocessing pipeline:
    #    - OneHotEncode categorical (drop='first' to avoid dummy trap)
    #    - Scale numeric (needed for Logistic Regression)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )

    # 6) Train-test split (stratified to preserve churn distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Fit preprocessing on train only (no leakage)
    preprocessor.fit(X_train)

    # Transform train/test
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save artifacts
    ensure_dir(ARTIFACT_DIR)
    save_pickle(preprocessor, SCALER_PATH)

    # Save final feature names (important for Streamlit + explainability)
    # Get feature names from ColumnTransformer
    ohe = preprocessor.named_transformers_["cat"]
    ohe_feature_names = list(ohe.get_feature_names_out(cat_cols))

    final_feature_names = num_cols + ohe_feature_names
    save_json(final_feature_names, FEATURE_COLS_PATH)

    print("\nPreprocessing complete.")
    print("Train shape:", X_train_processed.shape)
    print("Test shape :", X_test_processed.shape)
    print("Saved:", SCALER_PATH)
    print("Saved:", FEATURE_COLS_PATH)

    # (Optional) Save processed arrays as CSV later if needed
    # But for ML pipelines, saving transformer is best.


if __name__ == "__main__":
    main()