import pandas as pd
import os

# Input and Output paths
input_path = r'C:\Users\Asus\Desktop\exl training\exl-credit-churn-analysis\data\raw\exl_credit_card_churn_data.csv.csv'
output_path = r'C:\Users\Asus\Desktop\exl training\exl-credit-churn-analysis\data\processed\churn_cleaned.csv'

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

def clean_data(path):
    # Load the data
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # Remove whitespace in column names

    # Step 1: Show null values before cleaning
    print("Null values before cleaning:")
    print(df.isnull().sum())

    # Step 2: Fill nulls based on custom rules
    fill_values = {
        'Gender': df['Gender'].mode()[0],
        'Age': df['Age'].mean(),
        'Tenure': df['Tenure'].median(),
        'Balance': df['Balance'].mean(),
        'NumOfProducts': df['NumOfProducts'].mode()[0],
        'HasCrCard': df['HasCrCard'].mode()[0],
        'IsActiveMember': df['IsActiveMember'].mode()[0],
        'EstimatedSalary': df['EstimatedSalary'].mean(),
        'Churn': df['Churn'].mode()[0]
    }

    for col, value in fill_values.items():
        if df[col].isnull().sum() > 0:
            print(f"Filling {df[col].isnull().sum()} missing values in '{col}' with {value}")
            df[col].fillna(value, inplace=True)

    # Step 3: Show null values after filling
    print("\nNull values after cleaning:")
    print(df.isnull().sum())

    # Step 4: Remove duplicate rows
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"\nRemoved {before - after} duplicate rows.")

    # Step 5: Remove outliers using IQR method
    def remove_outliers(col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        print(f"\nOutlier detection for '{col}':")
        print(f"  - Q1 = {Q1:.2f}")
        print(f"  - Q3 = {Q3:.2f}")
        print(f"  - IQR = {IQR:.2f}")
        print(f"  - Lower Bound = {lower:.2f}")
        print(f"  - Upper Bound = {upper:.2f}")

        before_rows = df.shape[0]
        df_cleaned = df[(df[col] >= lower) & (df[col] <= upper)]
        after_rows = df_cleaned.shape[0]
        print(f"  - Removed {before_rows - after_rows} outliers from '{col}'")

        return df_cleaned

    numerical_cols = ['Age', 'Balance', 'EstimatedSalary']
    for col in numerical_cols:
        df = remove_outliers(col)

    # Step 6: Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")

    return df

if __name__ == "__main__":
    clean_data(input_path)
