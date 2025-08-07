import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# Input and Output paths
input_path = r'C:\Users\Asus\Desktop\exl training\exl-credit-churn-analysis\data\processed\churn_cleaned.csv'
output_path = r'C:\Users\Asus\Desktop\exl training\exl-credit-churn-analysis\data\processed\churn_final_encoded.csv'

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

def process_features(path):
    df = pd.read_csv(path)

    # Drop CustomerID if present
    if 'CustomerID' in df.columns:
        df.drop(columns=['CustomerID'], inplace=True)

    # One-Hot Encode Gender
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].astype(str).str.strip().str.lower()
        gender_dummies = pd.get_dummies(df['Gender'], prefix='Gender')
        df = pd.concat([df.drop(columns=['Gender']), gender_dummies], axis=1)

    # Convert 'Yes'/'No' or '1.0'/'0.0' to binary integers
    def convert_to_binary(col):
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].map({'yes': 1, 'no': 0, '1.0': 1, '0.0': 0, '1': 1, '0': 0})
        df[col] = df[col].fillna(0).astype(int)

    for col in ['HasCrCard', 'IsActiveMember', 'Churn']:
        if col in df.columns:
            convert_to_binary(col)

    # Feature Engineering
    df['BalanceToSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1e-6)
    df['TenureProductRatio'] = df['Tenure'] / (df['NumOfProducts'] + 1e-6)
    df['IsHighBalance'] = (df['Balance'] > df['EstimatedSalary']).astype(int)

    # Normalize numeric features
    scaler = MinMaxScaler()
    numeric_cols = [
        'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'EstimatedSalary', 'BalanceToSalaryRatio', 'TenureProductRatio'
    ]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Final check: Ensure Churn is binary int
    df = df[df['Churn'].isin([0, 1])]
    df['Churn'] = df['Churn'].astype(int)

    # Print feature summary
    print("\nFinal Feature Summary:")
    print(df.dtypes)
    print("\nUnique values per column:")
    print(df.nunique())
    print("\nValue ranges for normalized features:")
    print(df[numeric_cols].describe().T[['min', 'max']])

    # Save to output CSV
    df.to_csv(output_path, index=False)
    print(f"\nProcessed and encoded data saved to: {output_path}")

    return df

if __name__ == "__main__":
    process_features(input_path)
