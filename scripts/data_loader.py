import pandas as pd
import mysql.connector
import os

# --- Configuration ---
# Path to your raw CSV file
csv_path = r'C:\Users\Asus\Desktop\exl training\exl-credit-churn-analysis\data\raw\exl_credit_card_churn_data.csv.csv'

# Local MySQL database credentials
DB_USER = 'root'
DB_PASSWORD = 'Anend@2001'  # Replace with your password
DB_HOST = 'localhost'
DB_PORT = 3306
DB_NAME = 'exl_churn'
TABLE_NAME = 'customer_churndata'

# --- Function to load data and upload to MySQL ---
def load_and_upload_data():
    """
    Loads data from a CSV file into a Pandas DataFrame and then
    uploads it to a specified table in a local MySQL database
    using mysql-connector-python.
    """
    try:
        # Load data from the CSV file
        df = pd.read_csv(csv_path)
        print("Data loaded successfully from CSV.")
        print(df.head())

        # Establish a connection to the MySQL database
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
        cursor = conn.cursor()
        print("Database connection established.")

        # --- Drop the table if it exists to ensure a clean upload ---
        cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        print(f"Existing table '{TABLE_NAME}' dropped (if it existed).")

        # --- Create the table from the DataFrame columns and dtypes ---
        columns = df.columns
        column_defs = ", ".join([f"`{col}` VARCHAR(255)" for col in columns])
        create_table_query = f"CREATE TABLE {TABLE_NAME} ({column_defs})"
        cursor.execute(create_table_query)
        print(f"Table '{TABLE_NAME}' created.")

        # --- Prepare the INSERT query and upload data row by row ---
        insert_query = f"INSERT INTO {TABLE_NAME} ({', '.join([f'`{col}`' for col in columns])}) VALUES ({', '.join(['%s'] * len(columns))})"
        
        # Convert DataFrame rows to a list of tuples for insertion
        data_to_upload = [tuple(row) for row in df.values]
        
        # Execute the query to insert the data
        cursor.executemany(insert_query, data_to_upload)
        
        # Commit the changes to the database
        conn.commit()
        print(f"Data uploaded successfully to the '{TABLE_NAME}' table.")
    
    except FileNotFoundError:
        print(f"Error: The file at '{csv_path}' was not found.")
    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Close the connection and cursor
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    load_and_upload_data()