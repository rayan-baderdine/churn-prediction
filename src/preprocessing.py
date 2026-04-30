import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_clean(path):
    df = pd.read_csv(path)
    
    # Fix TotalCharges (whitespace strings → NaN → fill with 0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Drop customerID — not a feature
    df.drop('customerID', axis=1, inplace=True)
    
    # Binary encode target
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    return df

def encode_features(df):
    df = df.copy()
    
    # Binary columns: Yes/No → 1/0
    binary_cols = [
        'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'
    ]
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 
                               'No phone service': 0,
                               'No internet service': 0})
    
    # SeniorCitizen is already 0/1
    df['gender'] = (df['gender'] == 'Male').astype(int)
    
    # One-hot encode multi-category columns
    df = pd.get_dummies(df, columns=['InternetService', 
                                      'Contract', 
                                      'PaymentMethod'],
                        drop_first=False)
    return df

def get_train_test(df, test_size=0.2, random_state=42):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, 
        stratify=y,           # keeps churn ratio balanced
        random_state=random_state
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])
    
    return X_train, X_test, y_train, y_test, scaler