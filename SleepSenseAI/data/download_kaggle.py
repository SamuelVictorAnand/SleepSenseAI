"""Download Sleep Health and Lifestyle Dataset from Kaggle using kaggle API.
Requires:
  - pip install kaggle
  - Place your Kaggle API token `kaggle.json` in ~/.kaggle/kaggle.json OR set env vars KAGGLE_USERNAME and KAGGLE_KEY
Usage:
  python data/download_kaggle.py --dataset uom190346a/sleep-health-and-lifestyle-dataset --out data/sleep_health.csv
If you prefer manual download, visit the dataset page and place the CSV into data/ as sleep_health.csv
"""
import argparse, os, subprocess, sys

def main(dataset='uom190346a/sleep-health-and-lifestyle-dataset', out='data/sleep_health.csv'):
    # Try to use kaggle CLI to download dataset
    try:
        # Check kaggle CLI availability
        subprocess.run(['kaggle','--version'], check=True, stdout=subprocess.DEVNULL)
    except Exception as e:
        print("kaggle CLI not found. Please install it with: pip install kaggle", file=sys.stderr)
        raise

    # Create data dir
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # Use kaggle datasets download and unzip
    print(f"Downloading dataset {dataset} to temporary zip...")
    tmpzip = 'tmp_kaggle_dataset.zip'
    subprocess.run(['kaggle','datasets','download','-d', dataset, '-f', 'Sleep_health_and_lifestyle_dataset.csv', '-p', '.', '--unzip'], check=True)
    # The above command will download & unzip to current dir; move expected file if present
    possible_names = ['Sleep_health_and_lifestyle_dataset.csv', 'sleep_health_and_lifestyle_dataset.csv', 'sleep_health_and_lifestyle_dataset (1).csv']
    found = None
    for name in possible_names:
        if os.path.exists(name):
            os.replace(name, out)
            found = out
            break
    if found:
        print(f"Saved dataset as {out}")
    else:
        # try to find any csv in cwd
        for f in os.listdir('.'):
            if f.lower().endswith('.csv'):
                os.replace(f, out)
                found = out
                break
    if not found:
        raise FileNotFoundError('Could not find the CSV after downloading. Please download manually from Kaggle and place as data/sleep_health.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='uom190346a/sleep-health-and-lifestyle-dataset')
    parser.add_argument('--out', default='data/sleep_health.csv')
    args = parser.parse_args()
    main(args.dataset, args.out)
