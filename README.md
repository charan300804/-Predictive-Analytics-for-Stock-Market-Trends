# 📈 Predictive Analytics for Stock Market Trends

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&style=for-the-badge)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-F7931E?logo=scikit-learn&style=for-the-badge)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&style=for-the-badge)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Math-013243?logo=numpy&style=for-the-badge)](https://numpy.org/)

This project implements a stock price forecasting system using machine learning. It leverages Support Vector Regression (SVR) and historical stock price data, applies feature engineering and data preprocessing, and performs parameter tuning to predict market trends.

---

## 🚀 Key Features

- **📊 Advanced Feature Engineering**: Calculates simple moving averages (SMA), historical volatility, daily returns, and trading volume indexes.
- **🤖 Support Vector Regression (SVR)**: Trains and compares SVR models with RBF, Linear, and Polynomial kernels to identify the best fit.
- **📈 Dynamic Visualizations**: Interactive plotting showing historical trends, kernel forecasts, and error margins.
- **⚙️ Model Hyperparameter Tuning**: Utilizes Grid Search optimization to find the best `C`, `epsilon`, and `gamma` settings.

---

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Machine Learning**: Scikit-Learn
- **Data Engineering**: Pandas, NumPy
- **Data Visualization**: Matplotlib

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/charan300804/-Predictive-Analytics-for-Stock-Market-Trends.git
   cd -Predictive-Analytics-for-Stock-Market-Trends
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run prediction model:
   ```bash
   python app.py
   ```
