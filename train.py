"""train.py
Train a RandomForest model on the churn dataset and save the trained model.
Usage:
    python src/train.py --data data/customer_churn.csv --save models/model.pkl
"""
import argparse, os, joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from preprocess import load_data, basic_cleaning, encode_features, split_features_target

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to CSV data')
    p.add_argument('--save', default='models/model.pkl', help='Path to save model')
    return p.parse_args()

def main():
    args = parse_args()
    df = load_data(args.data)
    df = basic_cleaning(df)
    df = encode_features(df)
    X, y = split_features_target(df)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Handle imbalance with SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    # Scale numerical features
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    # Model and training
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    # Optional GridSearch (commented to keep runtime short)
    # param_grid = {'n_estimators': [100,200], 'max_depth':[None,10,20]}
    # gs = GridSearchCV(clf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    # gs.fit(X_train_res, y_train_res)
    # best = gs.best_estimator_
    clf.fit(X_train_res, y_train_res)
    preds = clf.predict(X_test_scaled)
    print('Accuracy:', accuracy_score(y_test, preds))
    print('Classification Report:\n', classification_report(y_test, preds))
    print('Confusion Matrix:\n', confusion_matrix(y_test, preds))
    # ROC-AUC (if probability available)
    try:
        proba = clf.predict_proba(X_test_scaled)[:,1]
        print('ROC AUC:', roc_auc_score(y_test, proba))
    except:
        pass
    # Save model + scaler
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    joblib.dump({'model': clf, 'scaler': scaler}, args.save)
    print('Saved model to', args.save)

if __name__ == '__main__':
    main()
