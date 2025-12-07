    import json, os

def save_feature_cols(cols, path='models/feature_columns.json'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,'w') as f:
        json.dump(cols, f)
    print('Saved feature columns to', path)
