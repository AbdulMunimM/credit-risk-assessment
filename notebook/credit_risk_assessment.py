# Credit Card Default Prediction using Machine Learning

## 1. Import Required Libraries
# ===============================
# Data Manipulation
# ===============================
import pandas as pd
import numpy as np

# ===============================
# Data Visualization
# ===============================
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Data Preprocessing
# ===============================
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# ===============================
# Machine Learning Models
# ===============================
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ===============================
# Hyperparameter Tuning
# ===============================
from sklearn.model_selection import GridSearchCV

# ===============================
# Model Evaluation
# ===============================
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# ===============================
# Save Model
# ===============================
import joblib

# ===============================
# Ignore Warnings
# ===============================
import warnings
warnings.filterwarnings("ignore")

## 2. Load Dataset

# Load the dataset
df = pd.read_csv("C:\\Users\\nudas\\ML Projects\\Credit Risk Assessment\\data\\credit_card_default.csv")

# Display first five records
df.head()

## 3. Dataset Overview"""

# Dataset dimensions
print(f"Dataset Shape: {df.shape}")

# Column names
df.columns

# Data types and missing values
df.info()

# Statistical summary
df.describe().T

"""## 4. Data Quality Check"""

# Missing values
df.isnull().sum()

# Duplicate records
print("Duplicate Records:", df.duplicated().sum())

# Remove duplicate records
df.drop_duplicates(inplace=True)

print("Dataset Shape:", df.shape)

"""## 5. Exploratory Data Analysis"""

# Target class distribution
df["default payment next month"].value_counts()

# Percentage distribution
(
    df["default payment next month"]
    .value_counts(normalize=True)
    * 100
)

plt.figure(figsize=(6,4))

sns.countplot(
    data=df,
    x="default payment next month"
)

plt.title("Credit Card Default Distribution")
plt.xlabel("Default Next Month")
plt.ylabel("Number of Customers")

plt.show()

# Categorical Features
for col in ["SEX", "EDUCATION", "MARRIAGE"]:
    print(f"\n{col}")
    print(df[col].value_counts().sort_index())

# Repayment Status Features
for col in ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]:
    print(f"\n{col}")
    print(df[col].value_counts().sort_index())

"""## 6. Correlation Analysis"""

correlation = df.corr(numeric_only=True)

correlation["default payment next month"].sort_values(
    ascending=False
)

plt.figure(figsize=(18,12))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation Heatmap")

plt.show()

"""## 7. Feature Engineering"""

# Rename target column
df.rename(
    columns={"default payment next month": "default"},
    inplace=True
)

# Remove ID column
df.drop(columns=["ID"], inplace=True)

df.head()

# Separate Features and Target

X = df.drop(columns=["default"])

y = df["default"]

print("Feature Matrix:", X.shape)
print("Target Vector :", y.shape)

"""## 8. Feature Categorization"""

# Numerical Features
numerical_features = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]

# Categorical Features
categorical_features = [
    "SEX",
    "EDUCATION",
    "MARRIAGE"
]

# Ordinal Features
ordinal_features = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]

print(f"Numerical Features : {len(numerical_features)}")
print(f"Categorical Features: {len(categorical_features)}")
print(f"Ordinal Features    : {len(ordinal_features)}")

"""## 9. Train-Test Split"""

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training Samples : {X_train.shape}")
print(f"Testing Samples  : {X_test.shape}")

print("Training Class Distribution")

print(
    y_train.value_counts(normalize=True) * 100
)

print("\nTesting Class Distribution")

print(
    y_test.value_counts(normalize=True) * 100
)

"""## 10. Data Preprocessing"""

# Standardize Numerical Features
numeric_transformer = StandardScaler()

# One-Hot Encode Categorical Features
categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

# Combine all preprocessing steps

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
        ("ord", "passthrough", ordinal_features)
    ]
)

"""## 11. Model Evaluation Function"""

def evaluate_model(model_name, y_true, y_pred, y_prob):
    """
    Evaluate a classification model using multiple performance metrics.
    """

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    print("=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    print("\nClassification Report")
    print(classification_report(y_true, y_pred))

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        cmap="Blues"
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

    # ROC Curve
    RocCurveDisplay.from_predictions(
        y_true,
        y_prob
    )

    plt.title(f"{model_name} - ROC Curve")
    plt.show()

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    }

"""## 12. Model Comparison Container"""

# Store evaluation results for all models

model_results = []

"""## 12. Baseline Model - Logistic Regression"""

# Logistic Regression Pipeline
logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        random_state=42,
        max_iter=1000
    ))
])

# Train the model
logistic_pipeline.fit(X_train, y_train)

# Predictions
y_pred_lr = logistic_pipeline.predict(X_test)
y_prob_lr = logistic_pipeline.predict_proba(X_test)[:, 1]

# Evaluate
lr_results = evaluate_model(
    "Logistic Regression",
    y_test,
    y_pred_lr,
    y_prob_lr
)

"""## 13. Baseline Model - Decision Tree"""

# Decision Tree Pipeline
decision_tree_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        random_state=42
    ))
])

# Train the model
decision_tree_pipeline.fit(X_train, y_train)

# Predictions
y_pred_dt = decision_tree_pipeline.predict(X_test)
y_prob_dt = decision_tree_pipeline.predict_proba(X_test)[:, 1]

# Evaluate
dt_results = evaluate_model(
    "Decision Tree",
    y_test,
    y_pred_dt,
    y_prob_dt
)

"""## 14. Baseline Model - Random Forest"""

# Random Forest Pipeline
random_forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42
    ))
])

# Train the model
random_forest_pipeline.fit(X_train, y_train)

# Predictions
y_pred_rf = random_forest_pipeline.predict(X_test)
y_prob_rf = random_forest_pipeline.predict_proba(X_test)[:, 1]

# Evaluate
rf_results = evaluate_model(
    "Random Forest",
    y_test,
    y_pred_rf,
    y_prob_rf
)

"""## 15. Baseline Model Comparison"""

comparison_df = pd.DataFrame([
    lr_results,
    dt_results,
    rf_results
])

comparison_df = comparison_df.round(4)

comparison_df

comparison_df.style.highlight_max(
    subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
    color="lightgreen"
)

"""## 16. Random Forest Hyperparameter Tuning"""

# Tuned Random Forest Pipeline
rf_tuned_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42,
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        n_jobs=-1
    ))
])

# Train the model
rf_tuned_pipeline.fit(X_train, y_train)

# Predictions
y_pred_rf_tuned = rf_tuned_pipeline.predict(X_test)
y_prob_rf_tuned = rf_tuned_pipeline.predict_proba(X_test)[:, 1]

# Evaluate
rf_tuned_results = evaluate_model(
    "Tuned Random Forest",
    y_test,
    y_pred_rf_tuned,
    y_prob_rf_tuned
)

"""## 17. Baseline vs Tuned Random Forest"""

rf_comparison = pd.DataFrame([
    rf_results,
    rf_tuned_results
]).round(4)

rf_comparison

rf_comparison.style.highlight_max(
    subset=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"
    ],
    color="lightgreen"
)

"""## 18. Model Comparison After Manual Tuning"""

comparison_df = pd.DataFrame([
    lr_results,
    dt_results,
    rf_results,
    rf_tuned_results
]).round(4)

comparison_df

comparison_df.style.highlight_max(
    subset=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"
    ],
    color="lightgreen"
)

"""## 19. Hyperparameter Optimization using GridSearchCV"""

# Random Forest Pipeline for Grid Search
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ))
])

"""## 20. Define Hyperparameter Search Space"""

param_grid = {
    "classifier__n_estimators": [200, 300],
    "classifier__max_depth": [10, 15],
    "classifier__min_samples_split": [5, 10],
    "classifier__min_samples_leaf": [2, 4],
    "classifier__class_weight": ["balanced"]
}

print(f"Total combinations: {2*2*2*2}")

"""## 21. Configure GridSearchCV"""

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=2
)

"""## 22. Train GridSearchCV"""

grid_search.fit(X_train, y_train)

"""## 23. Best Hyperparameters"""

print("Best Parameters\n")
print(grid_search.best_params_)

print("\nBest Cross Validation F1 Score")
print(round(grid_search.best_score_,4))

"""## 24. Evaluate Optimized Model"""

best_rf = grid_search.best_estimator_

y_pred_best = best_rf.predict(X_test)

y_prob_best = best_rf.predict_proba(X_test)[:,1]

grid_results = evaluate_model(
    "GridSearch Random Forest",
    y_test,
    y_pred_best,
    y_prob_best
)

"""## 25. Final Model Comparison"""

final_comparison = pd.DataFrame([
    lr_results,
    dt_results,
    rf_results,
    rf_tuned_results,
    grid_results
]).round(4)

final_comparison

final_comparison.sort_values(
    by="F1 Score",
    ascending=False
)

final_comparison.style.highlight_max(
    subset=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"
    ],
    color="lightgreen"
)

"""## 26. Final Model Selection"""

print("Selected Model : GridSearch Random Forest")
print("Reason:")
print("- Highest F1 Score")
print("- Highest ROC-AUC")
print("- Better Recall for identifying default customers")
print("- Uses optimized hyperparameters found through cross-validation")

"""## 27. Feature Importance Analysis"""

# Extract the trained Random Forest classifier
rf_model = best_rf.named_steps["classifier"]

# Extract the fitted preprocessor
preprocessor_fitted = best_rf.named_steps["preprocessor"]

"""## 28. Retrieve Processed Feature Names"""

# Numerical feature names
numeric_names = numerical_features

# Encoded categorical feature names
categorical_names = (
    preprocessor_fitted
    .named_transformers_["cat"]
    .get_feature_names_out(categorical_features)
)

# Ordinal feature names
ordinal_names = ordinal_features

# Combine all processed feature names
feature_names = (
    list(numeric_names)
    + list(categorical_names)
    + list(ordinal_names)
)

print(f"Total Processed Features: {len(feature_names)}")

"""## 29. Calculate Feature Importance"""

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.head(15)

"""## 30. Top 15 Most Important Features"""

top_features = feature_importance.head(15)

plt.figure(figsize=(10,7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.gca().invert_yaxis()

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Most Important Features")

plt.tight_layout()

plt.show()

"""## 31. Save the Final Model"""

import os
import joblib

# Go one folder up (Credit Risk Assessment)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Save model

from pathlib import Path
import joblib

# Get the project root directory (Credit Risk Assessment)
project_root = Path(__file__).resolve().parent.parent

# Path to the existing model folder
model_path = project_root / "model" / "credit_default_prediction_model.pkl"

# Save the trained model
joblib.dump(best_rf, model_path)

print("✅ Model saved successfully!")
print(f"📁 Saved to: {model_path}")
