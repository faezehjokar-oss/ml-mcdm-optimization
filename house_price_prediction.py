# House Price Prediction
# Faezeh Jokar - PhD Research

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load and prepare data
def prepare_data(df):
    df = df.dropna()
    X = df.drop('price', axis=1)
    y = df['price']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    return predictions
