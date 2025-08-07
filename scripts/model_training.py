import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# File paths
input_path = r'C:\Users\Asus\Desktop\exl training\exl-credit-churn-analysis\data\processed\churn_final_encoded.csv'
model_dir = r'C:\Users\Asus\Desktop\exl training\exl-credit-churn-analysis\model'
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'churn_model.pkl')
metrics_path = os.path.join(model_dir, 'model_metrics.txt')
confusion_matrix_path = os.path.join(model_dir, 'confusion_matrix.png')

# Load data
df = pd.read_csv(input_path)

if 'Churn' not in df.columns:
    raise ValueError("The 'Churn' column is missing from the dataset.")

# Features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model with best parameters
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    max_depth=10,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)
clf_report = classification_report(y_test, y_pred)

# Feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = feature_importances.sort_values(ascending=False).head(3)

# Save model
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# Save metrics to file
with open(metrics_path, 'w') as f:
    f.write("Model Evaluation Metrics\n")
    f.write("========================\n")
    f.write(f"Best Parameters: {model.get_params()}\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix))
    f.write("\n\nClassification Report:\n")
    f.write(clf_report)
    f.write("\nTop 3 Features by Importance:\n")
    f.write(str(top_features.to_string()))

# Show metrics in console
print("\nModel Evaluation Metrics")
print("========================")
print(f"Best Parameters: {model.get_params()}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(clf_report)
print("Top 3 Features by Importance:")
print(top_features.to_string())

# Plot and save confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(confusion_matrix_path)
plt.show()

print(f"\nConfusion matrix saved to: {confusion_matrix_path}")
