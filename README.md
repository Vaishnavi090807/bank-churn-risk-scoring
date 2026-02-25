#Bank Customer Churn – Predictive Modeling & Risk Scoring

#Project Overview
Customer churn is a major challenge for banks, as retaining existing customers is more cost-effective than acquiring new ones.  
This project builds an **end-to-end Machine Learning pipeline** to predict customer churn and assign an interpretable risk score.



##Project Structure

bank-churn-risk-scoring/
│
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── data/
│   └── European_Bank.xlsx     # Bank customer dataset
│
├── artifacts/
│   └── threshold_report.csv   # Threshold tuning results
│
├── reports/
│   ├── feature_importance_native.csv
│   ├── feature_importance_permutation.csv
│   └── shap_mean_abs.csv
│
└── src/
    ├── 01_preprocess.py       # Data preprocessing
    ├── 02_feature_engineering.py
    ├── 03_train_models.py     # Model training & evaluation
    ├── 04_explainability.py   # Model explainability
    ├── helpers.py
    ├── metrics.py
    ├── utils.py
    └── __init__.py

##Dataset Description
- **Format:** Excel (`.xlsx`)
- **Domain:** Banking
- **Target Variable:** `Exited`
  - `1` → Customer churned
  - `0` → Customer retained

##Key Features
- CreditScore  
- Geography  
- Gender  
- Age  
- Tenure  
- Balance  
- NumOfProducts  
- HasCrCard  
- IsActiveMember  
- EstimatedSalary  

##Feature Engineering
To enhance model performance, the following features were engineered:

- **Balance_to_Salary** – Balance relative to customer income  
- **Product_Density** – Number of products per year of relationship  
- **Engagement_Product_Interaction** – Interaction between activity and products  

##Machine Learning Models
The following models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

##Model Selection Strategy
- **Primary metric:** ROC-AUC  
- **Secondary objective:** Reduce false positives while maintaining recall  
- A tuned probability threshold was selected to optimize business impact.

##Model Explainability
To ensure interpretability and transparency:

- Native Feature Importance
- Permutation Importance
- SHAP-based explanations

All explainability outputs are saved in the `reports/` directory.

##Streamlit Web Application
The interactive Streamlit app allows users to:

- Input customer details
- Predict churn probability
- View risk score (0–100)
- Classify customers into Low / Medium / High risk
- Perform what-if scenario analysis
- Inspect top churn drivers

##Run the App
```bash
streamlit run app.py
