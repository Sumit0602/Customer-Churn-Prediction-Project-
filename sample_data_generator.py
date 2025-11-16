"""sample_data_generator.py
Generate a small synthetic customer churn CSV for testing.
"""
import pandas as pd
import numpy as np
import argparse

def gen(n=500, seed=42):
    np.random.seed(seed)
    genders = np.random.choice(['Male','Female'], n)
    senior = np.random.choice([0,1], n, p=[0.85,0.15])
    partner = np.random.choice(['Yes','No'], n)
    dependents = np.random.choice(['Yes','No'], n)
    tenure = np.random.randint(0,72,n)
    monthly = np.round(np.random.uniform(20,120,n),2)
    total = np.round(tenure * monthly + np.random.normal(0,50,n),2)
    contract = np.random.choice(['Month-to-month','One year','Two year'], n, p=[0.6,0.2,0.2])
    payment = np.random.choice(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'], n)
    internet = np.random.choice(['DSL','Fiber optic','No'], n, p=[0.4,0.45,0.15])
    churn = np.where((tenure < 12) & (contract == 'Month-to-month') & (np.random.rand(n) > 0.6), 'Yes', 'No')
    df = pd.DataFrame({
        'gender': genders,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
        'Contract': contract,
        'PaymentMethod': payment,
        'InternetService': internet,
        'Churn': churn
    })
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/customer_churn_sample.csv')
    p.add_argument('--rows', type=int, default=500)
    args = p.parse_args()
    df = gen(args.rows)
    df.to_csv(args.out, index=False)
    print('Saved sample to', args.out)

if __name__ == '__main__':
    main()
