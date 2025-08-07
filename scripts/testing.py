import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import os

# Model path
model_path = r'C:\Users\Asus\Desktop\exl training\exl-credit-churn-analysis\model\churn_model.pkl'

# Sample raw input (simulate real-life user entry)
raw_input = {
    'Age': 42,
    'Tenure': 3,
    'Balance': 75000,
    'NumOfProducts': 2,
    'HasCrCard': 'Yes',
    'IsActiveMember': 'No',
    'EstimatedSalary': 60000,
    'Gender': 'Female'
}

# Convert to DataFrame
df = pd.DataFrame([raw_input])

# Backup for displaying original input
original_input = df.copy()

# Step 1: Standardize Gender values
df['Gender'] = df['Gender'].astype(str).str.strip().str.lower()
df['Gender'] = df['Gender'].replace(['', 'nan', 'none'], pd.NA)
df['Gender'] = df['Gender'].fillna('female')  # default fallback

# Step 2: One-hot encode Gender
df = pd.get_dummies(df, columns=['Gender'])

# Ensure both gender columns exist
for col in ['Gender_female', 'Gender_male']:
    if col not in df.columns:
        df[col] = 0

# Step 3: Convert 'Yes'/'No' to 1/0
yes_no_map = {'yes': 1, 'no': 0}
df['HasCrCard'] = df['HasCrCard'].astype(str).str.lower().map(yes_no_map).fillna(0).astype(int)
df['IsActiveMember'] = df['IsActiveMember'].astype(str).str.lower().map(yes_no_map).fillna(0).astype(int)

# Step 4: Normalize numeric features
scaler = MinMaxScaler()
numeric_cols = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Step 5: Ensure correct column order
final_columns = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                 'IsActiveMember', 'EstimatedSalary', 'Gender_female', 'Gender_male']
for col in final_columns:
    if col not in df.columns:
        df[col] = 0
df = df[final_columns]

# Step 6: Load model and predict
with open(model_path, 'rb') as f:
    model = pickle.load(f)

prediction = model.predict(df)[0]
result = 'Churn' if prediction == 1 else 'Not Churn'

# Step 7: Display input and result
print("\n--- CUSTOMER INPUT ---")
print(original_input.to_string(index=False))

print("\n--- PREDICTION RESULT ---")
print(f"Prediction: {result}")
