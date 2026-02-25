import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utils import ensure_dir, save_pickle, save_json

# -----------------------------
# CONFIG (Exact paths)
# -----------------------------
DATA_PATH = "../data/European_Bank.csv"
ARTIFACT_DIR = "../artifacts"

PREPROCESSOR_PATH = f"{ARTIFACT_DIR}/preprocessor.pkl"
FEATURE_COLS_PATH = f"{ARTIFACT_DIR}/feature_columns.json"

TARGET_COL = "Exited"
DROP_COLS = ["CustomerId", "Surname"]  # non-informative


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived features required by the guide:
    1) Balance-to-Salary ratio
    2) Product density indicator
    3) Engagement-product interaction
    """
    # Safety to avoid division by zero
    eps = 1e-6

    # 1) Balance-to-Salary ratio
    df["Balance_to_Salary"] = df["Balance"] / (df["EstimatedSalary"] + eps)

    # 2) Product density indicator (simple + meaningful):
    #    products per year with bank (Tenure can be 0)
    df["Product_Density"] = df["NumOfProducts"] / (df["Tenure"] + 1)

    # 3) Engagement-product interaction:
    #    combines activity and number of products
    df["Engagement_Product_Interaction"] = df["IsActiveMember"] * df["NumOfProducts"]

    return df


def main():
    # 1) Load dataset
    df = pd.read_csv(DATA_PATH)
    print("Loaded:", df.shape)

    # 2) Missing value handling (usually none, but safe)
    if df.isnull().sum().sum() > 0:
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
        print("Missing values handled.")

    # 3) Drop non-informative features
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=c)

    # 4) Feature Engineering (Step 2)
    df = add_engineered_features(df)
    print("After feature engineering:", df.shape)

    # 5) Split X and y
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Identify categorical + numeric columns
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if X[c].dtype != "object"]

    print("\nCategorical columns:", cat_cols)
    print("Numeric columns:", num_cols)

    # 6) Train-test split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 7) Preprocessing pipeline:
    #    - Scale numeric
    #    - One-hot encode categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )

    # Fit only on train (no leakage)
    preprocessor.fit(X_train)

    # Transform (ready for models later)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 8) Save artifacts
    ensure_dir(ARTIFACT_DIR)
    save_pickle(preprocessor, PREPROCESSOR_PATH)

    # Save final column names (important for Streamlit + explainability)
    ohe = preprocessor.named_transformers_["cat"]
    ohe_feature_names = list(ohe.get_feature_names_out(cat_cols))
    final_feature_names = num_cols + ohe_feature_names
    save_json(final_feature_names, FEATURE_COLS_PATH)

    print("\nStep 2 complete (Feature Engineering + Preprocessing).")
    print("Train shape:", X_train_processed.shape)
    print("Test shape :", X_test_processed.shape)
    print("Saved:", PREPROCESSOR_PATH)
    print("Saved:", FEATURE_COLS_PATH)


if __name__ == "__main__":
    main()