import pandas as pd

# Load dataset using the second row as headers
df = pd.read_csv(r"C:\Users\nudas\ML Projects\Credit Risk Assessment\data\credit_card_default.csv", header=1)

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