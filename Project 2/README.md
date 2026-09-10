# 🚕 SmartRide NCR – Ride-Hailing Demand Category Classification

A machine learning project that classifies ride-hailing demand across Delhi NCR into **Low**, **Medium**, or **High** demand categories based on historical ride patterns, location, time, and previous demand.

Built using **Python, Streamlit, Scikit-learn, Pandas, and Folium** with an interactive dashboard for demand analysis, visualization, and prediction.

---

## 🎯 Problem Statement

Ride-hailing platforms need to anticipate demand across different locations and times to improve driver allocation and reduce passenger wait times.

This project predicts whether demand for a given scenario will be:

- Low Demand
- Medium Demand
- High Demand

using supervised machine learning.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Folium
- Joblib

---

## 🤖 Machine Learning Models Evaluated

The following classification algorithms were trained and compared:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

The best-performing model was selected and saved as:

```text
models/
└── demand_classifier.pkl
```

---

## 📂 Project Structure

```text
Project2/
│
├── app.py
├── run_pipeline.py
├── requirements.txt
├── README.md
│
├── data
│   ├── raw
│   │   ├── ncr_ride_bookings.csv
│   │   └── geocoded_locations.csv
│   │
│   └── processed
│       └── demand_dataset.csv
│
├── models
│   └── demand_classifier.pkl
│
├── outputs
│
└── src
    ├── __init__.py
    ├── preprocessing.py
    ├── feature_engineering.py
    ├── demand_creation.py
    ├── model_training.py
    ├── evaluation.py
    ├── predictor.py
    └── map_utils.py
```

---

## ⚙️ Project Workflow

```text
Raw Ride Booking Data
           │
           ▼
     Data Cleaning
           │
           ▼
  Feature Engineering
           │
           ▼
 Demand Category Creation
           │
           ▼
   Model Training & Evaluation
           │
           ▼
  Best Model Saved (.pkl)
           │
           ▼
 Streamlit Dashboard
           │
    ┌──────┴──────┐
    ▼             ▼
Demand Analysis  Demand Prediction
```

---

## 📊 Dashboard Features

### Demand Analysis

- Filter records by demand category
- Filter records by hour range
- KPI summary cards
- Interactive Delhi NCR demand map
- Filtered dataset table
- Demand distribution visualization

### Demand Prediction

Predict demand category using:

- Pickup Location
- Weekday
- Hour
- Previous Hour Demand (auto-detected)

Output:

- Low Demand
- Medium Demand
- High Demand

with business-friendly interpretation.

---

## 📈 Features Used for Prediction

- Hour
- Weekday
- Previous Demand
- Latitude
- Longitude

These features are transformed and passed to the trained classifier for prediction.

---

## 🚀 Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

---

## 📌 Project Outcome

This project demonstrates how machine learning can be used to classify ride demand patterns and support operational decisions such as driver allocation and demand monitoring across Delhi NCR.