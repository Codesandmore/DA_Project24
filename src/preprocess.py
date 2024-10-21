# preprocess.py

import pandas as pd

def load_data(file_path):
    try:
        return pd.read_csv(file_path, encoding='latin1')  # Adjust encoding as necessary
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame or handle error appropriately

def preprocess_data(data):
    """ print("Columns in DataFrame:", data.columns) """  # Debugging statement
    if 'v1' in data.columns:
        data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})
    else:
        print("Column 'v1' not found in the DataFrame.")
        # Handle the missing column case
    return data
