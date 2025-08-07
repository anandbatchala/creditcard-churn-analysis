import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle
import mysql.connector
from sklearn.preprocessing import MinMaxScaler

# Load trained model
model_path = r'C:\Users\Asus\Desktop\exl training\exl-credit-churn-analysis\model\churn_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# MySQL connection configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Anend@2001',
    'database': 'exl_churn'
}

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Step 1: Collect input
    raw_input = {
        'Age': request.form['Age'],
        'Tenure': request.form['Tenure'],
        'Balance': request.form['Balance'],
        'NumOfProducts': request.form['NumOfProducts'],
        'HasCrCard': request.form['HasCrCard'],
        'IsActiveMember': request.form['IsActiveMember'],
        'EstimatedSalary': request.form['EstimatedSalary'],
        'Gender': request.form['Gender']
    }

    original_input = raw_input.copy()

    # Step 2: Convert to DataFrame
    df = pd.DataFrame([raw_input])

    # Step 3: Standardize Gender
    df['Gender'] = df['Gender'].astype(str).str.strip().str.lower()
    df = pd.get_dummies(df, columns=['Gender'])

    # Ensure both Gender_female and Gender_male columns exist
    for col in ['Gender_female', 'Gender_male']:
        if col not in df.columns:
            df[col] = 0

    # Step 4: Convert Yes/No to binary
    yes_no_map = {'yes': 1, 'no': 0}
    df['HasCrCard'] = df['HasCrCard'].astype(str).str.lower().map(yes_no_map).fillna(0).astype(int)
    df['IsActiveMember'] = df['IsActiveMember'].astype(str).str.lower().map(yes_no_map).fillna(0).astype(int)

    # Step 5: Convert numeric columns
    numeric_cols = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Step 6: Feature engineering
    df['BalanceToSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1e-6)
    df['TenureProductRatio'] = df['Tenure'] / (df['NumOfProducts'] + 1e-6)
    df['IsHighBalance'] = (df['Balance'] > df['EstimatedSalary']).astype(int)

    # Step 7: Normalize numeric features
    scaler = MinMaxScaler()
    norm_cols = numeric_cols + ['BalanceToSalaryRatio', 'TenureProductRatio']
    df[norm_cols] = scaler.fit_transform(df[norm_cols])

    # Step 8: Ensure column order
    final_columns = [
        'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
        'EstimatedSalary', 'Gender_female', 'Gender_male',
        'BalanceToSalaryRatio', 'TenureProductRatio', 'IsHighBalance'
    ]

    for col in final_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[final_columns]

    # Step 9: Predict
    prediction = int(model.predict(df)[0])
    result = 'Churn' if prediction == 1 else 'Not Churn'

    # Step 10: Store in MySQL
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions 
        (age, tenure, balance, num_products, has_cr_card, is_active_member, estimated_salary, 
         gender_female, gender_male, balance_to_salary_ratio, tenure_product_ratio, is_high_balance, prediction)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        float(df['Age'].values[0]), 
        float(df['Tenure'].values[0]), 
        float(df['Balance'].values[0]),
        int(df['NumOfProducts'].values[0]), 
        int(df['HasCrCard'].values[0]), 
        int(df['IsActiveMember'].values[0]),
        float(df['EstimatedSalary'].values[0]), 
        int(df['Gender_female'].values[0]), 
        int(df['Gender_male'].values[0]),
        float(df['BalanceToSalaryRatio'].values[0]),
        float(df['TenureProductRatio'].values[0]),
        int(df['IsHighBalance'].values[0]),
        int(prediction)
    ))

    conn.commit()
    cursor.close()
    conn.close()

    return render_template('index.html', prediction=result, input_data=original_input)

if __name__ == '__main__':
    app.run(debug=True)
