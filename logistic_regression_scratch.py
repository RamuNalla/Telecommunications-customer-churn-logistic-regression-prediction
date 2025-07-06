import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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

for col in X_train.columns:
    if not pd.api.types.is_numeric_dtype(X_train[col]):
        print(f"Non-numeric column in training data: {col}, dtype: {X_train[col].dtype}")
        print(X_train[col].unique())
        raise TypeError(f"Column '{col}' is not numeric")

for col in X_test.columns:
    if not pd.api.types.is_numeric_dtype(X_test[col]):
        print(f"Non-numeric column in test data: {col}, dtype: {X_test[col].dtype}")
        print(X_test[col].unique())
        raise TypeError(f"Column '{col}' is not numeric")

# Ensure proper numeric conversion
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)

X_train_np = X_train.values                                                     # Convert DataFrames to NumPy arrays for scratch implementation
X_test_np = X_test.values

X_train_b = np.c_[np.ones((X_train_np.shape[0], 1)), X_train_np]                # Add a bias (intercept) term to X
X_test_b = np.c_[np.ones((X_test_np.shape[0], 1)), X_test_np]

print(f"\nProcessed Training data shape (with bias): {X_train_b.shape}, {y_train.shape}")
print(f"Processed Testing data shape (with bias): {X_test_b.shape}, {y_test.shape}")
print(X.head())

# ------------------------------------ 2. LOGISTIC REGRESSION FROM SCRATCH ----------------------------------


def sigmoid(z):                             # Sigmoid activation function.
    return 1 / (1 + np.exp(-z))

def hypothesis(X, theta):                   # Computes the hypothesis (predicted probabilities) for logistic regression.
    return sigmoid(X @ theta)

def cost_function(X, y, theta):             # Computes the Binary Cross-Entropy cost.
    m = len(y)
    predictions = hypothesis(X, theta)
    predictions = np.clip(predictions, 1e-10, 1 - 1e-10)                                        # Avoid log(0) by clipping predictions
    cost = (-1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

def gradient_descent(X, y, theta, learning_rate, n_iterations):                                 # Performs gradient descent to optimize theta for logistic regression.

    m = len(y)
    cost_history = []

    for iteration in range(n_iterations):
        predictions = hypothesis(X, theta)
        errors = predictions - y
        gradient = (1 / m) * X.T @ errors # Gradient calculation
        theta = theta - learning_rate * gradient # Update theta
        cost = cost_function(X, y, theta)
        cost_history.append(cost)

        if iteration % (n_iterations // 10) == 0:
            print(f"Iteration {iteration}/{n_iterations}, Cost: {cost:.4f}")

    return theta, cost_history


theta_scratch = np.zeros((X_train_b.shape[1], 1))               # Initialize weights to zeros
learning_rate = 0.05                                             # Learning rate might need tuning
n_iterations = 1000


theta_optimized, cost_history = gradient_descent(X_train_b, y_train, theta_scratch, learning_rate, n_iterations)    # Train the model


print(f"Optimized theta (scratch): {theta_optimized.flatten()}")
print(f"Final cost (scratch): {cost_function(X_train_b, y_train, theta_optimized):.4f}")


y_prob_scratch = hypothesis(X_test_b, theta_optimized)          # Make predictions on the test set. Get probabilities
y_pred_scratch = (y_prob_scratch >= 0.5).astype(int)            # Convert probabilities to binary predictions (threshold at 0.5)

# ------------------------------------ 3. EVALUATIONS FOR THE SCRATCH MODEL ----------------------------------

accuracy_s = accuracy_score(y_test, y_pred_scratch)
precision_s = precision_score(y_test, y_pred_scratch)
recall_s = recall_score(y_test, y_pred_scratch)
f1_s = f1_score(y_test, y_pred_scratch)
conf_matrix_s = confusion_matrix(y_test, y_pred_scratch)
roc_auc_s = roc_auc_score(y_test, y_prob_scratch)

print(f"Accuracy: {accuracy_s:.4f}")
print(f"Precision: {precision_s:.4f}")
print(f"Recall: {recall_s:.4f}")
print(f"F1-Score: {f1_s:.4f}")
print(f"ROC AUC Score: {roc_auc_s:.4f}")
print("\nConfusion Matrix (From Scratch):")
print(conf_matrix_s)
print("  [[True Negatives  False Positives]")
print("   [False Negatives True Positives]]")

# ------------------------------------ 4. PLOTTING ----------------------------------

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(range(n_iterations), cost_history)
plt.title('Cost Function History (From Scratch)')
plt.xlabel('Iterations')
plt.ylabel('Cost (Binary Cross-Entropy)')
plt.grid(True)

# Plotting a sample of predicted probabilities vs actual (difficult to visualize multi-dim data)
# Instead, let's visualize the distribution of probabilities for churned vs non-churned
plt.subplot(1, 2, 2)
sns.histplot(y_prob_scratch[y_test.flatten() == 0], color='blue', label='Non-Churn (0)', kde=True, stat='density', alpha=0.5)
sns.histplot(y_prob_scratch[y_test.flatten() == 1], color='red', label='Churn (1)', kde=True, stat='density', alpha=0.5)
plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Boundary (0.5)')
plt.title('Distribution of Predicted Probabilities (From Scratch)')
plt.xlabel('Predicted Probability of Churn')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ------------------------------------ 5. METRICS INTERPRETATION ----------------------------------

print(f"Accuracy ({accuracy_s:.4f}): Proportion of total predictions that were correct.")
print(f"Precision ({precision_s:.4f}): Out of all predicted churners, how many actually churned.")
print(f"Recall ({recall_s:.4f}): Out of all actual churners, how many did the model correctly identify.")
print(f"F1-Score ({f1_s:.4f}): Harmonic mean of Precision and Recall, useful for imbalanced classes.")
print("\nConfusion Matrix:")
print(f"  True Negatives ({conf_matrix_s[0,0]}): Correctly predicted non-churners.")
print(f"  False Positives ({conf_matrix_s[0,1]}): Incorrectly predicted churners (Type I error).")
print(f"  False Negatives ({conf_matrix_s[1,0]}): Incorrectly predicted non-churners (Type II error).")
print(f"  True Positives ({conf_matrix_s[1,1]}): Correctly predicted churners.")
