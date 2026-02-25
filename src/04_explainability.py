import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

from utils import ensure_dir, load_pickle, load_json

# -----------------------------
# CONFIG (Exact paths)
# -----------------------------
DATA_PATH = "../data/European_Bank.csv"
ARTIFACT_DIR = "../artifacts"
REPORTS_DIR = "../reports"

BEST_MODEL_PATH = f"{ARTIFACT_DIR}/best_model.pkl"
FEATURE_COLS_PATH = f"{ARTIFACT_DIR}/feature_columns.json"

TARGET_COL = "Exited"
DROP_COLS = ["CustomerId", "Surname"]  # non-informative


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    df["Balance_to_Salary"] = df["Balance"] / (df["EstimatedSalary"] + eps)
    df["Product_Density"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["Engagement_Product_Interaction"] = df["IsActiveMember"] * df["NumOfProducts"]
    return df


def save_bar_plot(names, values, title, out_path, top_n=20):
    # sort descending
    order = np.argsort(values)[::-1][:top_n]
    names = np.array(names)[order]
    values = np.array(values)[order]

    plt.figure()
    plt.barh(names[::-1], values[::-1])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ensure_dir(REPORTS_DIR)

    # 1) Load best model pipeline + feature names
    pipe = load_pickle(BEST_MODEL_PATH)
    feature_names = load_json(FEATURE_COLS_PATH)

    # pipeline steps
    preprocessor = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    # 2) Load data (same cleaning + feature engineering as earlier steps)
    df = pd.read_csv(DATA_PATH)

    if df.isnull().sum().sum() > 0:
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=c)

    df = add_engineered_features(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # stratified split (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 3) FEATURE IMPORTANCE (native if available)
    native_importance = None
    native_title = None

    if hasattr(model, "feature_importances_"):
        native_importance = model.feature_importances_
        native_title = "Native Feature Importance (Tree-Based)"
    elif hasattr(model, "coef_"):
        # Logistic Regression: use absolute coefficients
        coef = model.coef_.ravel()
        native_importance = np.abs(coef)
        native_title = "Native Feature Importance (|LogReg Coefficients|)"

    if native_importance is not None:
        out_csv = os.path.join(REPORTS_DIR, "feature_importance_native.csv")
        out_png = os.path.join(REPORTS_DIR, "feature_importance_native.png")

        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": native_importance
        }).sort_values("importance", ascending=False)

        imp_df.to_csv(out_csv, index=False)
        save_bar_plot(imp_df["feature"], imp_df["importance"],
                      native_title, out_png, top_n=20)

        print("Saved:", out_csv)
        print("Saved:", out_png)
    else:
        print("Native feature importance not available for this model.")

    # 4) PERMUTATION IMPORTANCE (works for any model; more reliable baseline)
    # NOTE: Use pipeline directly with raw X_test
    perm = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="roc_auc"
    )

    perm_df = pd.DataFrame({
        "feature": X_test.columns,  # raw features (before encoding)
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std
    }).sort_values("perm_importance_mean", ascending=False)

    out_csv = os.path.join(REPORTS_DIR, "feature_importance_permutation.csv")
    out_png = os.path.join(REPORTS_DIR, "feature_importance_permutation.png")

    perm_df.to_csv(out_csv, index=False)
    save_bar_plot(
        perm_df["feature"],
        perm_df["perm_importance_mean"],
        "Permutation Importance (ROC-AUC impact)",
        out_png,
        top_n=15
    )

    print("Saved:", out_csv)
    print("Saved:", out_png)

    # 5) SHAP (optional)
    # If shap isn't installed, code will skip safely.
    try:
        import shap

        # transform X_test to model input space
        X_test_trans = preprocessor.transform(X_test)

        # pick explainer based on model type
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_trans)
            # For binary, shap_values can be list [class0, class1]
            if isinstance(shap_values, list):
                shap_vals = shap_values[1]
            else:
                shap_vals = shap_values
        else:
            # linear fallback
            explainer = shap.LinearExplainer(model, X_test_trans)
            shap_vals = explainer.shap_values(X_test_trans)

        # summary plot
        shap_png = os.path.join(REPORTS_DIR, "shap_summary.png")
        plt.figure()
        shap.summary_plot(shap_vals, X_test_trans, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(shap_png, dpi=200)
        plt.close()

        print("Saved:", shap_png)

        # top features by mean |shap|
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        shap_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False)

        shap_csv = os.path.join(REPORTS_DIR, "shap_mean_abs.csv")
        shap_df.to_csv(shap_csv, index=False)

        print("Saved:", shap_csv)

    except Exception as e:
        print("SHAP skipped (not installed or incompatible).")
        print("Reason:", str(e))

    # 6) Partial Dependence Plots (PDP) on raw features (pipeline handles encoding)
    # Choose most business-meaningful features for churn
    pdp_features = ["Age", "Balance", "IsActiveMember", "NumOfProducts", "CreditScore"]

    # keep only features that exist in dataset
    pdp_features = [f for f in pdp_features if f in X.columns]
    # Convert numeric columns to float to avoid sklearn PDP future error
    for col in X_test.columns:
        if pd.api.types.is_integer_dtype(X_test[col]):
            X_test[col] = X_test[col].astype(float)

    if len(pdp_features) > 0:
        pdp_png = os.path.join(REPORTS_DIR, "pdp_plots.png")
        plt.figure()
        PartialDependenceDisplay.from_estimator(
            pipe,
            X_test,
            features=pdp_features,
            kind="average"
        )
        plt.tight_layout()
        plt.savefig(pdp_png, dpi=200)
        plt.close()

        print("Saved:", pdp_png)
    else:
        print("No PDP features found in dataset columns.")

    print("\nStep 4 Explainability complete. Check the reports/ folder.")


if __name__ == "__main__":
    main()