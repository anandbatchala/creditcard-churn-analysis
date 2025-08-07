import pandas as pd

# Load the cleaned or processed dataset
path = r'C:\Users\Asus\Desktop\exl training\exl-credit-churn-analysis\data\processed\churn_final_encoded.csv'

df = pd.read_csv(path)

# Check unique values in Churn
print("\n Datset After Encoding and Feature engineering \n")
print(df.head())
