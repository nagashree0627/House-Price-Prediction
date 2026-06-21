# 🏠 House Price Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting house prices using Machine Learning techniques. The objective is to build regression models that can estimate the price of a house based on various property features such as area, number of bedrooms, bathrooms, stories, parking spaces, furnishing status, and location-related amenities.

The project demonstrates the complete Data Science workflow including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, visualization, and business insights.

---

## 🎯 Problem Statement

Real estate buyers and sellers often rely on assumptions or outdated comparisons to estimate property values. This project aims to develop a predictive model that can accurately estimate house prices and identify the factors that most influence property value.

---

## 📂 Dataset

Dataset: Housing Prices Dataset

Source: Kaggle Housing Prices Dataset

The dataset contains information about various residential properties and their corresponding prices.

### Features Include:

* Area
* Bedrooms
* Bathrooms
* Stories
* Main Road Access
* Guest Room Availability
* Basement Availability
* Hot Water Heating
* Air Conditioning
* Parking Spaces
* Preferred Area
* Furnishing Status
* Price (Target Variable)

---

## 🛠 Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📊 Project Workflow

### 1. Data Loading & Exploration

* Loaded the dataset using Pandas.
* Inspected data structure and summary statistics.
* Checked dataset dimensions and missing values.

### 2. Data Cleaning

* Removed duplicate records.
* Handled missing values.
* Converted categorical variables into numerical format using One-Hot Encoding.

### 3. Exploratory Data Analysis (EDA)

* Analyzed house price distribution.
* Examined correlations between features.
* Identified key factors influencing house prices.

### 4. Model Building

Two regression models were trained and evaluated:

#### Linear Regression

* Simple baseline model.
* Interpretable and efficient.

#### Random Forest Regressor

* Ensemble learning model.
* Captures complex non-linear relationships.
* Achieved better prediction performance.

### 5. Model Evaluation

Models were evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## 📈 Visualizations

The project includes the following visualizations:

### House Price Distribution

Shows the distribution of property prices.

### Correlation Heatmap

Displays relationships between features and house prices.

### Actual vs Predicted Prices

Compares model predictions with actual house prices.

---

## 🔍 Key Insights

* House area is one of the strongest predictors of price.
* Bathrooms and number of stories significantly impact property value.
* Houses located in preferred areas tend to have higher prices.
* Random Forest Regressor outperformed Linear Regression in prediction accuracy.
* Property size often has a greater influence on price than the number of bedrooms.

---

## 🚀 Results

The machine learning models successfully predicted house prices based on property characteristics.

The Random Forest Regressor demonstrated superior performance and provided more accurate predictions compared to Linear Regression.

---

## 📁 Project Structure

```text
HousePricePrediction/

├── analysis.ipynb
├── Housing.csv
├── README.md
├── summary.pdf
│
└── charts/
    ├── price_distribution.png
    ├── correlation_heatmap.png
    └── actual_vs_predicted.png
```

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone <https://github.com/nagashree0627/House-Price-Prediction.git>
```

2. Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Open Jupyter Notebook:

```bash
jupyter notebook
```

4. Run all cells in `analysis.ipynb`.

---

## 📚 Learning Outcomes

Through this project, I gained hands-on experience in:

* Data Cleaning and Preprocessing
* Exploratory Data Analysis
* Feature Engineering
* Regression Modeling
* Model Evaluation
* Data Visualization
* Business Insight Generation

---

## 👩‍💻 Author

**Naga Shree Satya Haritha Pampana**

B.Tech CSE (AI & ML)
Sreyas Institute of Engineering and Technology

IIT Madras BS Degree in Data Science and Applications (Diploma Level)

GitHub: https://github.com/nagashree0627
