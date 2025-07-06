import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score

# ------------------------------------ 1. LOAD DATA AND EDA ----------------------------------

# Make sure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in the same directory
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
except FileNotFoundError:
    print("Error: CVS file not found.")

print(df.head())
print(f"\nOriginal DataFrame shape: {df.shape}")

df = df.drop('customerID', axis=1)                                                  # Drop Customer ID (not needed for prediction)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')             # 'TotalCharges' column currently an object type due to some empty strings. Convert to numeric, coercing errors to NaN.
df['TotalCharges'].fillna(0, inplace=True)                                          # For simplicity, we'll fill with 0 as these are likely new customers with no charges yet.

for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',    # Convert 'No internet service' and 'No phone service' to 'No' for consistency in binary features
             'StreamingTV', 'StreamingMovies', 'MultipleLines']:
    df[col] = df[col].replace('No internet service', 'No')
    if col == 'MultipleLines':
        df[col] = df[col].replace('No phone service', 'No')

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',      # Encode binary categorical features (Yes/No, Male/Female) to 0/1
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
               'StreamingMovies', 'PaperlessBilling', 'Churn', 'gender']
for col in binary_cols:
    if col == 'gender':
        df[col] = df[col].map({'Female': 0, 'Male': 1})
    else:
        df[col] = df[col].map({'No': 0, 'Yes': 1})

# One-hot encode other categorical features
categorical_cols = ['InternetService', 'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)                  # drop_first to avoid multicollinearity

X = df.drop('Churn', axis=1)                            # Define features (X) and target (y)
y = df['Churn'].values.reshape(-1, 1)                   # Ensure y is a column vector

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)               # Stratify for imbalanced target

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']                   # Feature Scaling. Standardize numerical features for Gradient Descent
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


# ------------------------------------ 2. LOGISTIC REGRESSION WITH SKLEARN ----------------------------------


model_sklearn = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)  # 'solver' specifies the algorithm to use for optimization. 'liblinear' is good for small datasets.  'lbfgs' is a good default for larger datasets.

model_sklearn.fit(X_train, y_train)

y_pred_sklearn = model_sklearn.predict(X_test)                  # Make predictions on the test set
y_prob_sklearn = model_sklearn.predict_proba(X_test)[:, 1]      # Probabilities for the positive class (churn=1)


# ------------------------------------ 3. EVALUATIONS FOR THE SKLEARN MODEL ----------------------------------

accuracy_s = accuracy_score(y_test, y_pred_sklearn)
precision_s = precision_score(y_test, y_pred_sklearn)
recall_s = recall_score(y_test, y_pred_sklearn)
f1_s = f1_score(y_test, y_pred_sklearn)
conf_matrix_s = confusion_matrix(y_test, y_pred_sklearn)
roc_auc_s = roc_auc_score(y_test, y_prob_sklearn)

print(f"Accuracy: {accuracy_s:.4f}")
print(f"Precision: {precision_s:.4f}")
print(f"Recall: {recall_s:.4f}")
print(f"F1-Score: {f1_s:.4f}")
print(f"ROC AUC Score: {roc_auc_s:.4f}")
print("\nConfusion Matrix (Scikit-learn):")
print(conf_matrix_s)
print("  [[True Negatives  False Positives]")
print("   [False Negatives True Positives]]")


# ------------------------------------ 4. PLOTTING ----------------------------------

# Plotting Confusion Matrix
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_s, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted No Churn', 'Predicted Churn'],
            yticklabels=['Actual No Churn', 'Actual Churn'])
plt.title('Confusion Matrix (Scikit-learn)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Plotting ROC Curve
plt.subplot(1, 2, 2)
fpr, tpr, thresholds = roc_curve(y_test, y_prob_sklearn)
plt.plot(fpr, tpr, color='orange', label=f'ROC curve (area = {roc_auc_s:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (Scikit-learn)')
plt.legend(loc="lower right")
plt.grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------ 5. METRICS INTERPRETATION ----------------------------------

print(f"Accuracy ({accuracy_s:.4f}): Proportion of total predictions that were correct.")
print(f"Precision ({precision_s:.4f}): Out of all customers predicted to churn, {precision_s*100:.2f}% actually churned.")
print(f"Recall ({recall_s:.4f}): Out of all customers who actually churned, the model correctly identified {recall_s*100:.2f}% of them.")
print(f"F1-Score ({f1_s:.4f}): A balanced measure of precision and recall. Useful when you need a balance between minimizing false positives and false negatives.")
print(f"ROC AUC Score ({roc_auc_s:.4f}): Measures the ability of the model to distinguish between positive and negative classes. A higher AUC indicates a better model.")
print("\nConfusion Matrix:")
print(f"  True Negatives ({conf_matrix_s[0,0]}): Correctly predicted non-churners.")
print(f"  False Positives ({conf_matrix_s[0,1]}): Incorrectly predicted churners (Type I error). These are customers who did NOT churn but the model thought they would.")
print(f"  False Negatives ({conf_matrix_s[1,0]}): Incorrectly predicted non-churners (Type II error). These are actual churners that the model missed.")
print(f"  True Positives ({conf_matrix_s[1,1]}): Correctly predicted churners.")



