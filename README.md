# Customer Churn Prediction (Internship Project)

This repository contains code, documentation and scripts for an internship-level project:
**Customer Churn Prediction Using Machine Learning** (Python).

## Contents
- `src/` - Python source code (preprocessing, training, prediction)
- `data/` - Place your dataset here (`customer_churn.csv`). A small sample generator is provided.
- `models/` - Trained model artifacts will be saved here (`models/model.pkl`)
- `Customer_Churn_Training_Report_Final.docx` - Project report (you can edit student details)
- `requirements.txt` - Python dependencies
- `README.md` - This file
- `.gitignore` - Common files to ignore

## Quickstart (run locally)
1. Create a Python virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Prepare dataset:
- Put your `customer_churn.csv` into the `data/` folder.
- If you don't have a dataset, you can generate a small synthetic sample:
```bash
python src/sample_data_generator.py --out data/customer_churn_sample.csv --rows 500
```

3. Preprocess, train, and save model:
```bash
python src/train.py --data data/customer_churn.csv --save models/model.pkl
```

4. Predict on a single sample (example):
```bash
python src/predict.py --model models/model.pkl --input "gender=Male,SeniorCitizen=0,Partner=No,Dependents=No,tenure=12,MonthlyCharges=70.35,TotalCharges=843.5,Contract=Month-to-month,PaymentMethod=Electronic check,InternetService=DSL"
```

## Files of interest
- `src/preprocess.py` - Data cleaning and encoding pipeline
- `src/train.py` - Model training, evaluation and saving
- `src/predict.py` - Load model and predict for a single sample string
- `src/sample_data_generator.py` - Small CSV generator for testing

## Notes
- The provided code assumes a CSV with a `Churn` column (values 'Yes'/'No' or 1/0).
- Edit `src/config` variables as needed.
- The report DOCX contains the formatted project report; edit student details manually as you requested.

---
