"""
Credit Risk Assessment Project
Step 1 - Load and Inspect Dataset
"""

import pandas as pd

# ==========================================================
# Load Dataset
# ==========================================================

DATA_PATH = "data/credit_card_default.csv"

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.\n")

# ==========================================================
# Basic Information
# ==========================================================

print("=" * 60)
print("Dataset Shape")
print("=" * 60)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# ==========================================================
# Display First Rows
# ==========================================================

print("\n" + "=" * 60)
print("First Five Rows")
print("=" * 60)

print(df.head())

# ==========================================================
# Column Names
# ==========================================================

print("\n" + "=" * 60)
print("Column Names")
print("=" * 60)

for col in df.columns:
    print(col)

# ==========================================================
# Dataset Information
# ==========================================================

print("\n" + "=" * 60)
print("Dataset Information")
print("=" * 60)

print(df.info())

# ==========================================================
# Statistical Summary
# ==========================================================

print("\n" + "=" * 60)
print("Statistical Summary")
print("=" * 60)

print(df.describe(include="all"))

# ==========================================================
# Missing Values
# ==========================================================

print("\n" + "=" * 60)
print("Missing Values")
print("=" * 60)

print(df.isnull().sum())

# ==========================================================
# Duplicate Records
# ==========================================================

duplicates = df.duplicated().sum()

print("\n" + "=" * 60)
print("Duplicate Records")
print("=" * 60)

print(f"Duplicate Rows: {duplicates}")

print("\nInspection Complete.")

# Drop ID column
df.drop("ID", axis=1, inplace=True)

# Rename target column
df.rename(
    columns={
        "default payment next month": "Default"
    },
    inplace=True
)

X = df.drop("Default", axis=1)
y = df["Default"]

# ==========================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

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

