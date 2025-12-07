# Sleep Quality Prediction — Machine Learning Project

A simple and effective machine learning project that predicts Sleep Quality using real lifestyle and health-related data from the Sleep Health and Lifestyle dataset on Kaggle.

This project comes with complete, ready-to-run scripts for data preprocessing, model training, evaluation, and prediction.

# Features

✔️ Real Kaggle dataset

✔️ Cleaned & fully preprocessed ML pipeline

✔️ Logistic Regression model

✔️ Automatic saving of trained model + scaler

✔️ Single-sample or batch CSV predictions

✔️ Simple and modular Python scripts

# Dataset Setup (Required)

Download the dataset from Kaggle:

Dataset Link:
https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

After downloading:

Rename the downloaded file to:
sleep_health.csv

Place it inside the project directory:

data/sleep_health.csv

# Install Dependencies

Install all required libraries:

pip install -r requirements.txt

# Train the Model

Run the training script:

python src/train.py --data data/sleep_health.csv --out models


This will:

Preprocess the dataset

Train a Logistic Regression model

Save the model + scaler inside the models/ directory

# Make Predictions
1️⃣ Predict using a single input sample
python src/predict.py --input_sample "steps=8000,screen_time_hours=3.5,stress_level=2"

2️⃣ Predict using a CSV file
python src/predict.py --input_csv data/sample_to_predict.csv

