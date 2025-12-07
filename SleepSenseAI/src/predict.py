"""Prediction script that loads saved model and scaler and predicts on provided CSV or sample.
This version dynamically matches feature columns saved in models/feature_columns.json
Usage:
  python src/predict.py --input_csv data/sample_to_predict.csv
"""
import argparse, os, joblib, json, pandas as pd, numpy as np

def main(model_path='models/model.joblib', scaler_path='models/scaler.joblib', feature_path='models/feature_columns.json', input_csv=None, input_sample=None):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(feature_path):
        raise FileNotFoundError('Model, scaler or feature_columns.json not found. Run src/train.py first.')
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(feature_path,'r') as f:
        features = json.load(f)
    if input_csv:
        df = pd.read_csv(input_csv)
    elif input_sample:
        parts = [p.strip() for p in input_sample.split(',') if '=' in p]
        d = {}
        for p in parts:
            k,v = p.split('=',1)
            d[k.strip()] = float(v)
        df = pd.DataFrame([d])
    else:
        raise ValueError('Provide --input_csv or --input_sample')
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError('Missing columns in input: ' + ','.join(missing))
    X = df[features]
    X = X.fillna(X.median())
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:,1] if hasattr(model,'predict_proba') else None
    out = df.copy()
    out['predicted_sleep_quality'] = np.where(preds==1,'Good','Bad')
    if probs is not None:
        out['good_prob'] = np.round(probs,3)
    print(out.to_string(index=False))
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str)
    parser.add_argument('--input_sample', type=str)
    parser.add_argument('--model', default='models/model.joblib')
    parser.add_argument('--scaler', default='models/scaler.joblib')
    parser.add_argument('--features', default='models/feature_columns.json')
    args = parser.parse_args()
    main(args.model, args.scaler, args.features, args.input_csv, args.input_sample)
