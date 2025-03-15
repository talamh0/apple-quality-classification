import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load dataset
url = "https://raw.githubusercontent.com/rjdp07/Apple_Quality/main/apple_quality.csv"
df = pd.read_csv(url)
print(df.head())

# Handle missing values
df_cleaned = df.dropna()

# Extract features and target variables
X = df_cleaned.drop(columns=['A_id', 'Quality'])
y = df_cleaned['Quality'].map({'good': 1, 'bad': 0})
print("Target distribution:")
print(y.value_counts())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic regression without regularization
model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Logistic regression with L2 regularization
regularization_strengths = [0.1, 1, 10]
best_model, best_accuracy, best_C = None, 0, None
metrics_comparison = []

for C in regularization_strengths:
    model_l2 = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=1000)
    model_l2.fit(X_train_scaled, y_train)
    y_pred_l2 = model_l2.predict(X_test_scaled)
    
    accuracy_l2 = accuracy_score(y_test, y_pred_l2)
    precision_l2 = precision_score(y_test, y_pred_l2, zero_division=1)
    recall_l2 = recall_score(y_test, y_pred_l2, zero_division=1)
    f1_l2 = f1_score(y_test, y_pred_l2, zero_division=1)
    roc_auc_l2 = roc_auc_score(y_test, y_pred_l2)
    
    metrics_comparison.append((C, accuracy_l2, precision_l2, recall_l2, f1_l2, roc_auc_l2))
    
    if accuracy_l2 > best_accuracy:
        best_accuracy = accuracy_l2
        best_model = model_l2
        best_C = C

# Compare models
print("\nComparison of Models with and without Regularization:")
print(f"Baseline Model - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
for C, acc, prec, rec, f1, auc in metrics_comparison:
    print(f"Regularized Model (C={C}) - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}, ROC-AUC: {auc:.4f}")

# Plot ROC curve for best model
if best_model:
    y_scores = best_model.decision_function(X_test_scaled)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    best_auc = roc_auc_score(y_test, best_model.predict(X_test_scaled))
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (C={best_C}, AUC={best_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Best Model')
    plt.legend()
    plt.show()
