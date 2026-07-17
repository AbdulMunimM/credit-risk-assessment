import pandas as pd

# Load dataset using the second row as headers
df = pd.read_csv("/content/drive/MyDrive/Pandas/credit_card_default.csv", header=1)

# Drop ID column
df = df.drop(columns=["ID"])

# Rename target column
df = df.rename(columns={
    "default payment next month": "Default"
})

# Convert all columns to numeric
df = df.apply(pd.to_numeric)

# Verify
print(df.head())
print(df.info())
print("Dataset Shape:", df.shape)
print(df.dtypes)
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
df.shape
df.describe()
print(df["Default"].value_counts())
print(df["Default"].value_counts(normalize=True) * 100)
corr = df.corr()

corr["Default"].sort_values(ascending=False)
# Features
X = df.drop("Default", axis=1)

# Target
y = df["Default"]

print("X shape:", X.shape)
print("y shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("Training Features:", X_train.shape)
print("Testing Features :", X_test.shape)

print("Training Labels  :", y_train.shape)
print("Testing Labels   :", y_test.shape)
print("Training set:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting set:")
print(y_test.value_counts(normalize=True) * 100)
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")
# ==========================================================
# TARGET VARIABLE DISTRIBUTION
# ==========================================================

print("=" * 60)
print("Target Variable Distribution")
print("=" * 60)

print(df["Default"].value_counts())

print("\nPercentage Distribution")

print(df["Default"].value_counts(normalize=True) * 100)

plt.figure(figsize=(6,5))

sns.countplot(
    x="Default",
    data=df,
    palette="viridis"
)

plt.title("Default Payment Distribution")
plt.xlabel("Default")
plt.ylabel("Count")

plt.show()
# ==========================================================
# AGE DISTRIBUTION
# ==========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    df["AGE"],
    bins=30,
    kde=True
)

plt.title("Age Distribution")

plt.xlabel("Age")

plt.show()
# ==========================================================
# CREDIT LIMIT DISTRIBUTION
# ==========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    df["LIMIT_BAL"],
    bins=30,
    kde=True
)

plt.title("Credit Limit Distribution")

plt.xlabel("Credit Limit")

plt.show()
# ==========================================================
# GENDER DISTRIBUTION
# ==========================================================

gender = df["SEX"].replace({
    1: "Male",
    2: "Female"
})

plt.figure(figsize=(6,5))

sns.countplot(
    x=gender
)

plt.title("Gender Distribution")

plt.xlabel("Gender")

plt.show()
# ==========================================================
# EDUCATION DISTRIBUTION
# ==========================================================

education = df["EDUCATION"].replace({
    0: "Unknown",
    1: "Graduate",
    2: "University",
    3: "High School",
    4: "Others",
    5: "Unknown",
    6: "Unknown"
})

plt.figure(figsize=(8,5))

sns.countplot(
    x=education,
    order=education.value_counts().index
)

plt.xticks(rotation=20)

plt.title("Education Distribution")

plt.show()
# ==========================================================
# MARRIAGE DISTRIBUTION
# ==========================================================

marriage = df["MARRIAGE"].replace({
    0: "Unknown",
    1: "Married",
    2: "Single",
    3: "Others"
})

plt.figure(figsize=(6,5))

sns.countplot(
    x=marriage,
    order=marriage.value_counts().index
)

plt.title("Marriage Distribution")

plt.show()
# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

plt.figure(figsize=(18,12))

sns.heatmap(
    df.corr(),
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")

plt.show()
# ==========================================================
# FEATURE CORRELATION WITH TARGET
# ==========================================================

corr = df.corr()["Default"].sort_values(
    ascending=False
)

print(corr)
# ==========================================================
# IDENTIFY FEATURE TYPES
# ==========================================================

categorical_features = [
    "EDUCATION",
    "MARRIAGE"
]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

print("=" * 60)
print("Categorical Features")
print("=" * 60)

print(categorical_features)

print("\nNumerical Features")

print(numerical_features)
# ==========================================================
# DATA PREPROCESSING
# ==========================================================

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)

print("Preprocessor created successfully.")
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
# ==========================================================
# LOGISTIC REGRESSION PIPELINE
# ==========================================================

lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

print("Logistic Regression Pipeline Created")
# ==========================================================
# TRAIN LOGISTIC REGRESSION
# ==========================================================

print("=" * 60)
print("Training Logistic Regression...")
print("=" * 60)

lr_pipeline.fit(X_train, y_train)

print("Training Completed Successfully.")
# ==========================================================
# MAKE PREDICTIONS
# ==========================================================

y_pred_lr = lr_pipeline.predict(X_test)

# Probability predictions
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]
y_pred_lr
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
# ==========================================================
# LOGISTIC REGRESSION EVALUATION
# ==========================================================

accuracy = accuracy_score(y_test, y_pred_lr)
precision = precision_score(y_test, y_pred_lr)
recall = recall_score(y_test, y_pred_lr)
f1 = f1_score(y_test, y_pred_lr)
roc_auc = roc_auc_score(y_test, y_prob_lr)

print("=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print("\nClassification Report")
print("=" * 60)

print(classification_report(y_test, y_pred_lr))
# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
# ==========================================================
# ROC CURVE
# ==========================================================

RocCurveDisplay.from_predictions(
    y_test,
    y_prob_lr
)

plt.title("ROC Curve - Logistic Regression")
plt.show()
# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

feature_names = lr_pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

coefficients = lr_pipeline.named_steps[
    "model"
].coef_[0]

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

coef_df["Absolute"] = coef_df["Coefficient"].abs()

coef_df = coef_df.sort_values(
    by="Absolute",
    ascending=False
)

print(coef_df.head(15))
from sklearn.tree import DecisionTreeClassifier

# ==========================================================
# DECISION TREE PIPELINE
# ==========================================================

dt_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        DecisionTreeClassifier(
            random_state=42
        )
    )
])

print("Decision Tree Pipeline Created")
# ==========================================================
# TRAIN DECISION TREE
# ==========================================================

print("=" * 60)
print("Training Decision Tree...")
print("=" * 60)

dt_pipeline.fit(X_train, y_train)

print("Training Completed Successfully.")
# ==========================================================
# MAKE PREDICTIONS
# ==========================================================

y_pred_dt = dt_pipeline.predict(X_test)

y_prob_dt = dt_pipeline.predict_proba(X_test)[:, 1]
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
roc_auc_dt = roc_auc_score(y_test, y_prob_dt)

print("=" * 60)
print("DECISION TREE RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy_dt:.4f}")
print(f"Precision: {precision_dt:.4f}")
print(f"Recall   : {recall_dt:.4f}")
print(f"F1 Score : {f1_dt:.4f}")
print(f"ROC-AUC  : {roc_auc_dt:.4f}")
from sklearn.ensemble import RandomForestClassifier
# ==========================================================
# RANDOM FOREST PIPELINE
# ==========================================================

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    )
])

print("Random Forest Pipeline Created")
# ==========================================================
# TRAIN RANDOM FOREST
# ==========================================================

print("=" * 60)
print("Training Random Forest...")
print("=" * 60)

rf_pipeline.fit(X_train, y_train)

print("Training Completed Successfully.")
# ==========================================================
# MAKE PREDICTIONS
# ==========================================================

y_pred_rf = rf_pipeline.predict(X_test)

y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]
# ==========================================================
# RANDOM FOREST EVALUATION
# ==========================================================

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)

print("=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall   : {recall_rf:.4f}")
print(f"F1 Score : {f1_rf:.4f}")
print(f"ROC-AUC  : {roc_auc_rf:.4f}")
print("\nClassification Report")
print("=" * 60)

print(classification_report(y_test, y_pred_rf))
# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

feature_names = rf_pipeline.named_steps["preprocessor"].get_feature_names_out()

importances = rf_pipeline.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df.head(15))
plt.figure(figsize=(10,6))

sns.barplot(
    data=importance_df.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Important Features - Random Forest")

plt.show()
# ==========================================================
# MODEL COMPARISON
# ==========================================================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy,
        accuracy_dt,
        accuracy_rf
    ],
    "Precision": [
        precision,
        precision_dt,
        precision_rf
    ],
    "Recall": [
        recall,
        recall_dt,
        recall_rf
    ],
    "F1 Score": [
        f1,
        f1_dt,
        f1_rf
    ],
    "ROC-AUC": [
        roc_auc,
        roc_auc_dt,
        roc_auc_rf
    ]
})

results = results.sort_values(
    by="ROC-AUC",
    ascending=False
)

print(results)
import joblib

joblib.dump(
    rf_pipeline,
    "credit_risk_model.pkl"
)

print("Model saved.")
