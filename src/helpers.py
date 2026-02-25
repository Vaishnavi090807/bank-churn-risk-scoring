import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    MUST match Step 2/3/4 feature engineering exactly.
    """
    eps = 1e-6
    df["Balance_to_Salary"] = df["Balance"] / (df["EstimatedSalary"] + eps)
    df["Product_Density"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["Engagement_Product_Interaction"] = df["IsActiveMember"] * df["NumOfProducts"]
    return df


def risk_score(prob: float) -> int:
    """Convert probability to 0-100 score."""
    return int(round(prob * 100))


def risk_label(prob: float, threshold: float) -> str:
    """
    Simple interpretable labels.
    - High: prob >= threshold
    - Medium: 0.5*threshold <= prob < threshold
    - Low: below that
    """
    if prob >= threshold:
        return "HIGH"
    if prob >= (0.5 * threshold):
        return "MEDIUM"
    return "LOW"