"""Train script for Sleep Quality Prediction using real dataset.
Usage:
  python src/train.py --data data/sleep_health.csv --out models
If dataset missing, you can run: python data/download_kaggle.py
"""
import argparse, os, subprocess, pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_and_prepare(path):
    df = pd.read_csv(path)
    # Basic cleaning: lowercase columns
    df.columns = [c.strip() for c in df.columns]
    # Inspect: we expect a column named 'Quality of Sleep' or similar; try to find label
    label_candidates = [c for c in df.columns if 'quality' in c.lower() or 'sleep' in c.lower() and 'quality' in c.lower()]
    # Fallback to 'Quality of Sleep' or 'Quality' or 'sleep_quality'
    if 'Quality of Sleep' in df.columns:
        label_col = 'Quality of Sleep'
    elif 'Quality' in df.columns:
        label_col = 'Quality'
    elif 'sleep_quality' in df.columns:
        label_col = 'sleep_quality'
    elif len(label_candidates)>0:
        label_col = label_candidates[0]
    else:
        raise ValueError('Could not find label column. Please ensure dataset has a sleep quality column (Quality of Sleep or similar). Columns found: ' + ','.join(df.columns))
    # For safety, map textual qualities to Good/Bad: values like 'Good','Poor','Average' -> map.
    df = df.rename(columns={label_col:'sleep_quality'})
    df['sleep_quality'] = df['sleep_quality'].astype(str).str.strip()
    # Map common variants to Good/Bad
    good = set(['good','very good','excellent','normal','fair','0?'])
    bad = set(['poor','bad','very poor','insomnia','disorder','1?'])
    df['sleep_quality_label'] = df['sleep_quality'].str.lower().apply(lambda x: 1 if any(g in x for g in good) else (0 if any(b in x for b in bad) else (1 if 'good' in x else 0)))
    # Select feature columns: numeric ones commonly present
    feature_cols = [c for c in df.columns if c.lower() in ['steps','screen_time_hours','caffeine_mg','exercise_minutes','stress_level','bedtime_hour','sleep duration','sleep_duration','daily steps','daily_steps','age','physical activity']]
    # If not found, fallback to numeric columns except label
    if len(feature_cols)==0:
        feature_cols = [c for c in df.select_dtypes(include=[float,int]).columns if c not in ['sleep_quality_label']]
    X = df[feature_cols].copy()
    y = df['sleep_quality_label']
    # Fill numeric nans with median
    X = X.fillna(X.median())
    return X, y, feature_cols

def main(data_path='data/sleep_health.csv', out_dir='models'):
    if not os.path.exists(data_path):
        print('Dataset not found at', data_path)
        print('Please run: python data/download_kaggle.py  OR manually place dataset at data/sleep_health.csv')
        return
    X, y, feature_cols = load_and_prepare(data_path)
    print('Using features:', feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {acc:.4f}')
    print('Classification Report:\n', classification_report(y_test, y_pred))
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir,'model.joblib'))
        # Save feature columns
        import json
        with open(os.path.join(out_dir,'feature_columns.json'),'w') as f:
            json.dump(feature_cols, f)
        print('Saved feature columns to', os.path.join(out_dir,'feature_columns.json'))
    joblib.dump(scaler, os.path.join(out_dir,'scaler.joblib'))
    with open(os.path.join(out_dir,'evaluation.txt'),'w') as f:
        f.write(f'accuracy: {acc}\n\n')
        f.write(classification_report(y_test, y_pred))
    print('Saved model and scaler to', out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/sleep_health.csv')
    parser.add_argument('--out', default='models')
    args = parser.parse_args()
    main(args.data, args.out)
