"""predict.py
Simple CLI to load saved model and predict churn for a single sample provided as comma-separated key=value pairs.

Example:
python src/predict.py --model models/model.pkl --input "gender=Male,SeniorCitizen=0,Partner=No,Dependents=No,tenure=12,MonthlyCharges=70.35,TotalCharges=843.5,Contract=Month-to-month,PaymentMethod=Electronic check,InternetService=DSL"
"""
import argparse, joblib, pandas as pd
from preprocess import load_data, basic_cleaning, encode_features, split_features_target

def parse_input_to_df(sample_str, reference_df_path=None):
    # sample_str: "k1=v1,k2=v2,..."
    items = [s.strip() for s in sample_str.split(',') if s.strip()]
    d = {}
    for it in items:
        if '=' not in it: continue
        k,v = it.split('=',1)
        d[k.strip()] = v.strip()
    df = pd.DataFrame([d])
    # If reference_df_path provided, read it to get columns and types
    if reference_df_path:
        ref = pd.read_csv(reference_df_path)
        # Align columns - fill missing with defaults
        for c in ref.columns:
            if c not in df.columns:
                # fill numeric with 0, object with 'Unknown'
                df[c] = 0 if ref[c].dtype != object else 'Unknown'
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--input', required=True, help='Sample as comma separated key=value')
    p.add_argument('--ref', default=None, help='Optional reference CSV to align columns/types')
    args = p.parse_args()
    obj = joblib.load(args.model)
    model = obj['model']
    scaler = obj.get('scaler', None)
    sample_df = parse_input_to_df(args.input, args.ref)
    # Basic cleaning & encoding using same functions (best effort)
    from preprocess import basic_cleaning, encode_features
    sample_df = basic_cleaning(sample_df)
    sample_df = encode_features(sample_df)
    # If scaler exists, ensure columns align order with training features might differ.
    # This is a simple approach: select numeric columns and scale them.
    X = sample_df
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)
            proba = model.predict_proba(X_scaled)[:,1] if hasattr(model,'predict_proba') else None
        except Exception as e:
            # If transform fails, fallback to predict without scaling
            pred = model.predict(X)
            proba = None
    else:
        pred = model.predict(X)
        proba = None
    print('Prediction (1=Churn,0=No Churn):', pred)
    if proba is not None:
        print('Probability of churn:', proba)

if __name__ == '__main__':
    main()
