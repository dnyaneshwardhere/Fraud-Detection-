import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Set up base paths
BASE_DIR   = r"E:\PCCOE\Semesters\6th\ML\Sagar ML Mini Project"
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'fraud_transactions_data.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load data
df = pd.read_csv(DATA_PATH)

# Features and labels
X = df[['TransactionAmount']]
y = df['IsFraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to evaluate
models = {
    "LogisticRegression": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

# Store results
results = []

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec = recall_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100
    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) * 100
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Precision: {prec:.2f}%")
    print(f"Recall: {rec:.2f}%")
    print(f"F1-Score: {f1:.2f}%")
    print(f"ROC-AUC: {roc:.2f}%")
    print(f"ConfusionMatrix: {cm.tolist()}")

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc
    })

# Convert to DataFrame for plotting
results_df = pd.DataFrame(results)

# 🔥 Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.show()

plt.figure(figsize=(12, 6))
results_df.set_index('Model')[['Precision', 'Recall', 'F1-Score', 'ROC-AUC']].plot(kind='bar')
plt.title("Performance Metrics by Model")
plt.ylabel("Score (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
