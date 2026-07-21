# Credit Card Default Prediction

An end-to-end Machine Learning project that predicts whether a credit card customer will default on their next payment. The project covers data preprocessing, exploratory data analysis, model training, hyperparameter optimization, model evaluation, and deployment using Streamlit.

---

## Project Overview

Financial institutions must identify customers who are likely to default so they can make informed lending decisions and reduce financial risk.

This project predicts the target variable:

> **default**
>
> - `0` → Customer will not default
> - `1` → Customer is likely to default

The final model is deployed as an interactive Streamlit application.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

## Project Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Data Preprocessing
6. Baseline Model Training
7. Model Evaluation
8. Hyperparameter Optimization
9. Feature Importance Analysis
10. Model Deployment

---

## Data Preprocessing

The following preprocessing steps were applied:

- Removed duplicate records
- Removed the **ID** column
- Renamed the target column from:

```
default payment next month
```

to

```
default
```

- Train/Test Split (80:20)
- Stratified sampling to preserve class distribution

### Feature Processing

### Numerical Features

- StandardScaler

### Categorical Features

- OneHotEncoder (`handle_unknown="ignore"`)

### Ordinal Features

- Passed through without scaling

A Scikit-learn **Pipeline** and **ColumnTransformer** were used to prevent data leakage and ensure consistent preprocessing during both training and inference.

---

## Models Trained

The following baseline models were evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

After comparing their performance, Random Forest was selected for further optimization.

---

## Hyperparameter Optimization

Two optimization approaches were explored.

### 1. Manual Tuning

The Random Forest model was manually tuned using:

- `n_estimators = 300`
- `max_depth = 15`
- `min_samples_split = 10`
- `min_samples_leaf = 4`
- `class_weight = "balanced"`

This improved the model's ability to identify default customers.

### 2. GridSearchCV

A GridSearchCV search was then performed using **5-fold Cross Validation**.

### Search Space

| Hyperparameter | Values |
|---------------|--------|
| n_estimators | 200, 300 |
| max_depth | 10, 15 |
| min_samples_split | 5, 10 |
| min_samples_leaf | 2, 4 |
| class_weight | balanced |

The optimization objective was:

- **F1 Score**

The final deployed model is the **GridSearchCV Optimized Random Forest**.

---

## Model Evaluation

The models were evaluated using multiple classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve

---

## Why Accuracy Alone Is Not Enough

Credit default prediction is an **imbalanced classification problem**, where non-default customers significantly outnumber default customers.

A model may achieve high accuracy simply by predicting most customers as **non-default**, while failing to identify customers who are actually at risk.

For this reason, accuracy alone is not sufficient for evaluating credit risk models.

More informative metrics include:

### Precision

Measures how many predicted defaulters are actually defaulters.

Higher precision reduces unnecessary rejection of reliable customers.

### Recall

Measures how many actual defaulters are correctly identified.

Higher recall helps financial institutions detect more risky customers and reduce potential losses.

### F1 Score

Balances precision and recall.

Since both false approvals and false rejections carry financial consequences, F1 Score provides a more reliable measure of model performance than accuracy alone.

### ROC-AUC

Evaluates how well the model distinguishes between default and non-default customers across different decision thresholds.

---

## Final Model Selection

Although another model produced slightly higher accuracy, the **GridSearchCV Optimized Random Forest** was selected because it provided a better balance between precision and recall.

The final model was chosen because it achieved:

- Better F1 Score
- Better Recall
- Strong ROC-AUC
- Better generalization through 5-fold Cross Validation

These characteristics make it more suitable for real-world credit risk assessment than selecting a model based solely on accuracy.

---

## Feature Importance

The trained Random Forest model was used to compute feature importance scores.

The project also visualizes the **Top 15 Most Important Features**, providing insight into which variables contribute most to predicting customer default.

---

## Deployment

The trained model is serialized using Joblib and deployed with Streamlit.

Users can:

- Enter customer information
- Predict default risk
- View prediction confidence
- Receive an easy-to-understand risk assessment

---

## Repository Structure

```
Credit-Risk-Assessment/
│
├── app/
│   └── app.py
│
├── data/
│   └── credit_card_default.csv
│
├── model/
│   └── credit_default_prediction_model.pkl
│
├── notebook/
│   └── credit_risk_training.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Future Improvements

- Probability threshold optimization
- XGBoost and LightGBM comparison
- SHAP explainability
- Model monitoring
- Automated retraining pipeline