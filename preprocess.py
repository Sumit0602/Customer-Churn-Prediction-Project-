"""preprocess.py
Functions to load, clean and encode the customer churn dataset.
Expected target column: 'Churn' (Yes/No or 1/0).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def load_data(path):
    df = pd.read_csv(path)
    return df

def basic_cleaning(df):
    # Trim column names
    df.columns = [c.strip() for c in df.columns]
    # Convert TotalCharges to numeric (if present)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        imputer = SimpleImputer(strategy='median')
        df['TotalCharges'] = imputer.fit_transform(df[['TotalCharges']])
    # Drop customerID if present
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    return df

def encode_features(df, target='Churn'):
    df = df.copy()
    le = LabelEncoder()
    # Ensure target is binary 0/1
    if target in df.columns:
        if df[target].dtype == object:
            df[target] = le.fit_transform(df[target])
    # Encode all other object columns
    for col in df.select_dtypes(include=['object']).columns:
        if col == target: 
            continue
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def split_features_target(df, target='Churn'):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
