import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set(style="whitegrid")

# File paths
csv_path = r'C:\Users\Asus\Desktop\exl training\exl-credit-churn-analysis\data\raw\exl_credit_card_churn_data.csv.csv'
output_dir = r'C:\Users\Asus\Desktop\exl training\exl-credit-churn-analysis\scripts\visualizations'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the data
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()  # Remove whitespace in column names

# Function to save plots
def save_plot(fig, name):
    fig.savefig(os.path.join(output_dir, name), bbox_inches='tight')
    plt.close(fig)

# 1. Gender Distribution
fig = plt.figure(figsize=(6,4))
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
save_plot(fig, 'gender_distribution.png')

# 2. Age Distribution
fig = plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=10, kde=True)
plt.title('Age Distribution')
save_plot(fig, 'age_distribution.png')

# 3. Tenure Distribution
fig = plt.figure(figsize=(6,4))
sns.countplot(x='Tenure', data=df)
plt.title('Tenure Distribution')
save_plot(fig, 'tenure_distribution.png')

# 4. Balance Distribution
fig = plt.figure(figsize=(6,4))
sns.histplot(df['Balance'], bins=10, kde=True)
plt.title('Balance Distribution')
save_plot(fig, 'balance_distribution.png')

# 5. Estimated Salary Distribution
fig = plt.figure(figsize=(6,4))
sns.histplot(df['EstimatedSalary'], bins=10, kde=True)
plt.title('Estimated Salary Distribution')
save_plot(fig, 'salary_distribution.png')

# 6. Churn Count
fig = plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Count')
save_plot(fig, 'churn_count.png')

# 7. Gender vs Churn
fig = plt.figure(figsize=(6,4))
sns.countplot(x='Gender', hue='Churn', data=df)
plt.title('Gender vs Churn')
save_plot(fig, 'gender_vs_churn.png')

# 8. Age vs Churn
fig = plt.figure(figsize=(6,4))
sns.boxplot(x='Churn', y='Age', data=df)
plt.title('Age vs Churn')
save_plot(fig, 'age_vs_churn.png')

# 9. Tenure vs Churn
fig = plt.figure(figsize=(6,4))
sns.countplot(x='Tenure', hue='Churn', data=df)
plt.title('Tenure vs Churn')
save_plot(fig, 'tenure_vs_churn.png')

# 10. Balance vs Churn
fig = plt.figure(figsize=(6,4))
sns.boxplot(x='Churn', y='Balance', data=df)
plt.title('Balance vs Churn')
save_plot(fig, 'balance_vs_churn.png')

# 11. NumOfProducts vs Churn
fig = plt.figure(figsize=(6,4))
sns.countplot(x='NumOfProducts', hue='Churn', data=df)
plt.title('NumOfProducts vs Churn')
save_plot(fig, 'products_vs_churn.png')

# 12. HasCrCard vs Churn
fig = plt.figure(figsize=(6,4))
sns.countplot(x='HasCrCard', hue='Churn', data=df)
plt.title('Has Credit Card vs Churn')
save_plot(fig, 'crcard_vs_churn.png')

# 13. IsActiveMember vs Churn
fig = plt.figure(figsize=(6,4))
sns.countplot(x='IsActiveMember', hue='Churn', data=df)
plt.title('Active Member vs Churn')
save_plot(fig, 'active_vs_churn.png')

# 14. Estimated Salary vs Churn
fig = plt.figure(figsize=(6,4))
sns.boxplot(x='Churn', y='EstimatedSalary', data=df)
plt.title('Salary vs Churn')
save_plot(fig, 'salary_vs_churn.png')

# 15. Correlation Heatmap
fig = plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Matrix')
save_plot(fig, 'correlation_matrix.png')

print(f"✅ All visualizations saved in:\n{output_dir}")
