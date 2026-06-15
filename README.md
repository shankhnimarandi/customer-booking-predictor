# ✈️ Proactive Customer Acquisition: Airline Booking Predictor

## 📌 Project Overview
In the competitive airline industry, waiting for customers to purchase flights at the airport or runway is a reactive, losing strategy. This project implements an end-to-end machine learning pipeline built to proactively predict whether a customer will complete a holiday booking during their online search cycle. 

By leveraging a dataset of 50,000 customer sessions, this predictive framework allows marketing teams to target high-probability buyers with dynamic incentives before they close their sessions or cross-shop with competitors.

## 🚀 Key Features & Advanced ML Techniques
This pipeline stands out by tackling common real-world data engineering hurdles directly:
* **Zero Data Leakage:** Categorical encoding transformations are applied strictly *after* data splitting to preserve validation integrity.
* **High-Cardinality Target Encoding:** Deployed `TargetEncoder` on variables like `route` (700+ values) and `booking_origin` to capture geographical conversion rates without blowing up matrix sparsity.
* **Class Imbalance Mitigation:** Resolved a heavily skewed target variable (~15% baseline conversion rate) utilizing `StratifiedKFold` cross-validation and balanced class weighting.
* **Hyperparameter Optimization:** Utilized `GridSearchCV` to mathematically tune tree depth, split boundaries, and estimators to maximize the model's area under the ROC curve.

## 📊 Model Performance
* **Primary Metric:** Optimized to achieve a final **Test ROC AUC Score of ~0.78**.
* **Classification Framework:** Balanced to maximize precision-recall across the minority class, ensuring the business minimizes wasted marketing ad spend on false-positive targets.

## 🗺️ Key Business Insights
The Random Forest feature importance extraction yielded critical commercial conclusions:
1. **Geography Over Schedule:** A customer's country of origin (`booking_origin`) and chosen flight paths (`route`) are the absolute heaviest drivers of final conversions. Behavioral timelines like departure hours or specific days of the week have negligible predictive impact.
2. **The Booking Window:** The advance planning duration (`purchase_lead`) is highly ranked, offering a clear signal for when automated incentive pipelines should trigger.

## 🛠️ Tech Stack & Dependencies
* **Language:** Python
* **Libraries:** Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
