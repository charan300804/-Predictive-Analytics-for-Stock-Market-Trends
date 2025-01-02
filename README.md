# -Predictive-Analytics-for-Stock-Market-Trends


This project implements a stock price prediction model using Support Vector Regression (SVR). It leverages historical stock data, processes the data, performs feature engineering, and tunes model parameters for accurate predictions.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
- [Feature Engineering](#feature-engineering)
- [Model Evaluation](#model-evaluation)
- [Visualization](#visualization)
- [Acknowledgements](#acknowledgements)

## Introduction
The purpose of this project is to build a model that can predict stock prices based on historical data. The model uses SVR, which is known for its effectiveness in regression tasks. The project involves data preprocessing, feature engineering, parameter tuning, and evaluation.

## Installation
To run this project, you need to have Python and the following libraries installed:
- numpy
- matplotlib
- yfinance
- scikit-learn
- shap

You can install these libraries using pip:
```bash
pip install numpy matplotlib yfinance scikit-learn shap

## Usage
Clone the repository:

git clone https://github.com/charan300804/stock-price-prediction.git
## Navigate to the project directory:

cd stock-price-prediction

## Run the main script:

python app.py

## Functions
download_data(companies, start_date, end_date): Downloads historical stock data for the given companies within the specified date range.

preprocess_data(data): Preprocesses the data by filling missing values, capping outliers, and normalizing the 'Close' prices.

feature_engineering(data): Adds new features to the data, such as moving averages, momentum, volatility, and RSI.

parameter_tuning(data, companies): Tunes the SVR model parameters using GridSearchCV and evaluates the model.

evaluate_model(model, X, y, company): Evaluates the model using cross-validation and various metrics, and plots the results.

plot_results(y_test, y_pred, company): Plots the actual vs. predicted stock prices.

residual_analysis(y_test, y_pred, company): Analyzes the residuals of the model.

plot_feature_importance(model, X_test): Plots the feature importance using SHAP values.

### Feature Engineering
Feature engineering involves creating additional features from the existing data to improve the predictive power of the model. In this project, the following features are created:

10_MA: 10-day moving average of the 'Close' prices.

100_MA: 100-day moving average of the 'Close' prices.

Momentum: Difference between the current 'Close' price and the 'Close' price 10 days ago.

Volatility: Standard deviation of the 'Close' prices over a 10-day window.

RSI: Relative Strength Index, calculated over a 14-day window.

## Model Evaluation
The SVR model is evaluated using the following metrics:

Root Mean Squared Error (RMSE)

Mean Squared Error (MSE)

R-squared (R²)

Mean Absolute Error (MAE)

## Visualization
The project includes various visualizations to help understand the model's performance:

Actual vs. Predicted Stock Prices

Residuals Over Time

Feature Importance using SHAP values

## Acknowledgements
The yfinance library for providing easy access to historical stock data.

The scikit-learn library for powerful machine learning tools.

The shap library for interpreting machine learning models.
