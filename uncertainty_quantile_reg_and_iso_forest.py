# Databricks notebook source
# MAGIC %pip install xgboost

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd

table_name = "default.california_housing_train" 

housing_df = spark.table(table_name).toPandas()

y = housing_df['median_house_value'].values
X = housing_df.drop('median_house_value', axis=1).values

X_train_raw, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)
X_test  = scaler.transform(X_test_raw)

# COMMAND ----------

model_mean = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=100, 
    learning_rate=0.1,
    max_depth=5,
    n_jobs=-1
)
model_mean.fit(X_train, y_train)

# COMMAND ----------

print("2. Training Uncertainty Models (XGBoost Quantiles)...")

model_lower = xgb.XGBRegressor(
    objective='reg:quantileerror', 
    quantile_alpha=0.05,
    n_estimators=100, 
    learning_rate=0.1,
    max_depth=5,
    n_jobs=-1
)
model_lower.fit(X_train, y_train)

model_upper = xgb.XGBRegressor(
    objective='reg:quantileerror', 
    quantile_alpha=0.95,
    n_estimators=100, 
    learning_rate=0.1,
    max_depth=5,
    n_jobs=-1
)
model_upper.fit(X_train, y_train)

# COMMAND ----------

iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
iso_forest.fit(X_train)

# COMMAND ----------

def scan_house_uncertainty(X_new, model_mean, model_lower, model_upper, iso_forest):
    """
    Ties together Price Prediction, Interval Width (Noise), and Anomaly Detection (Trust).
    """
    pred_price = model_mean.predict(X_new)
    lower_bound = model_lower.predict(X_new)
    upper_bound = model_upper.predict(X_new)
    
    # Calculate Uncertainty Width (Aleatoric Risk)
    interval_width = upper_bound - lower_bound
    
    # Get Anomaly Score (Epistemic Risk / Trust) where -1 = Outlier, 1 = Normal
    # decision_function gives continuous score: Negative is "Weird", Positive is "Normal"
    # We flip it (-1 *) so Higher Score = More Alien/Weird
    weirdness_score = -1 * iso_forest.decision_function(X_new)
    is_anomaly = iso_forest.predict(X_new) # Returns -1 or 1

    # Define Thresholds (Tune these based on business risk)
    # Example: If weirdness > 0.05, it's too alien.
    # Example: If interval > $200k, it's too volatile.
    
    return {
        "Predicted_Price": pred_price,
        "Safe_Range": (lower_bound, upper_bound),
        "Range_Width": interval_width,
        "Weirdness_Score": weirdness_score,
        "Is_Anomaly": is_anomaly == -1
    }

# COMMAND ----------

results = scan_house_uncertainty(X_test[:5], model_mean, model_lower, model_upper, iso_forest)

# COMMAND ----------

for i in range(5):
    p = results["Predicted_Price"][i]
    w = results["Range_Width"][i]
    is_weird = results["Is_Anomaly"][i]
    score = results["Weirdness_Score"][i]
    
    print(f"House #{i}: Est. ${p:,.0f}")
    print(f"  > Market Uncertainty (Width): ${w:,.0f}")
    print(f"  > Trust Check: {'Uncommon HOUSE' if is_weird else 'Normal'} (Score: {score:.2f})")
    print("-" * 30)

# COMMAND ----------

def evaluate_uncertainty(y_true):
    # Coverage (PICP): What % of true values fell inside the interval?
    # We want this to be very close to 0.90 (since we used alpha 0.05 and 0.95)
    lower = model_lower.predict(X_test)
    upper = model_upper.predict(X_test)

    captured = (y_true >= lower) & (y_true <= upper)
    coverage = np.mean(captured)
    
    # Sharpness (MPIW): Average width of the interval
    # We want this to be as small as possible (while keeping coverage > 0.90)
    width = np.mean(upper - lower)

    avg_price = np.mean(y_true) 
    
    print(f"--- UNCERTAINTY REPORT CARD ---")
    print(f"Target Coverage:      90.0%")
    print(f"Actual Coverage:      {coverage*100:.1f}%  <-- {'✅ Good' if abs(coverage-0.9)<0.05 else '⚠️ Needs Calibration'}")
    print(f"Avg Interval Width:   ${width:,.0f}")
    print(f"Avg House Price:      ${avg_price:,.0f}")
    print(f"Relative Uncertainty: {width/avg_price*100:.1f}% of price")

# Run the evaluation
evaluate_uncertainty(y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

from sklearn.cluster import KMeans

X_spatial = X_train[:, [6, 7]] 

# 2. Train the Clusterer
# n_clusters=50 is a good starting point (creates 50 "neighborhoods")
kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
kmeans.fit(X_spatial)

# 3. Create the New Feature
# We don't replace Lat/Lon, we ADD this as a new column
train_clusters = kmeans.predict(X_train[:, [6, 7]])
test_clusters  = kmeans.predict(X_test[:, [6, 7]])

# 4. Reshape for concatenation
train_clusters = train_clusters.reshape(-1, 1)
test_clusters  = test_clusters.reshape(-1, 1)

# 5. Add to features
X_train_enhanced = np.hstack((X_train, train_clusters))
X_test_enhanced  = np.hstack((X_test, test_clusters))

# COMMAND ----------

model_mean_v2 = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=100, 
    learning_rate=0.1,
    max_depth=5,
    n_jobs=-1
)
model_mean_v2.fit(X_train_enhanced, y_train)

print("2. Training Uncertainty Models (XGBoost Quantiles)...")

model_lower_v2 = xgb.XGBRegressor(
    objective='reg:quantileerror', 
    quantile_alpha=0.05,
    n_estimators=100, 
    learning_rate=0.1,
    max_depth=5,
    n_jobs=-1
)
model_lower_v2.fit(X_train_enhanced, y_train)

model_upper_v2 = xgb.XGBRegressor(
    objective='reg:quantileerror', 
    quantile_alpha=0.95,
    n_estimators=100, 
    learning_rate=0.1,
    max_depth=5,
    n_jobs=-1
)
model_upper_v2.fit(X_train_enhanced, y_train)

iso_forest_v2 = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
iso_forest_v2.fit(X_train_enhanced)

# COMMAND ----------

def scan_house_uncertainty(X_new):
    """
    Ties together Price Prediction, Interval Width (Noise), and Anomaly Detection (Trust).
    """
    pred_price = model_mean_v2.predict(X_new)
    lower_bound = model_lower_v2.predict(X_new)
    upper_bound = model_upper_v2.predict(X_new)
    
    # Calculate Uncertainty Width (Aleatoric Risk)
    interval_width = upper_bound - lower_bound
    
    # Get Anomaly Score (Epistemic Risk / Trust) where -1 = Outlier, 1 = Normal
    # decision_function gives continuous score: Negative is "Weird", Positive is "Normal"
    # We flip it (-1 *) so Higher Score = More Alien/Weird
    weirdness_score = -1 * iso_forest_v2.decision_function(X_new)
    is_anomaly = iso_forest_v2.predict(X_new) # Returns -1 or 1

    # Define Thresholds (Tune these based on business risk)
    # Example: If weirdness > 0.05, it's too alien.
    # Example: If interval > $200k, it's too volatile.
    
    return {
        "Predicted_Price": pred_price,
        "Safe_Range": (lower_bound, upper_bound),
        "Range_Width": interval_width,
        "Weirdness_Score": weirdness_score,
        "Is_Anomaly": is_anomaly == -1
    }

results = scan_house_uncertainty(X_test_enhanced[:5])

for i in range(5):
    p = results["Predicted_Price"][i]
    w = results["Range_Width"][i]
    is_weird = results["Is_Anomaly"][i]
    score = results["Weirdness_Score"][i]
    
    print(f"House #{i}: Est. ${p:,.0f}")
    print(f"  > Market Uncertainty (Width): ${w:,.0f}")
    print(f"  > Trust Check: {'Uncommon HOUSE' if is_weird else 'Normal'} (Score: {score:.2f})")
    print("-" * 30)

def evaluate_uncertainty(y_true):
    # Coverage (PICP): What % of true values fell inside the interval?
    # We want this to be very close to 0.90 (since we used alpha 0.05 and 0.95)
    lower = model_lower_v2.predict(X_test_enhanced)
    upper = model_upper_v2.predict(X_test_enhanced)

    captured = (y_true >= lower) & (y_true <= upper)
    coverage = np.mean(captured)
    
    # Sharpness (MPIW): Average width of the interval
    # We want this to be as small as possible (while keeping coverage > 0.90)
    width = np.mean(upper - lower)

    avg_price = np.mean(y_true) 
    
    print(f"--- UNCERTAINTY REPORT CARD ---")
    print(f"Target Coverage:      90.0%")
    print(f"Actual Coverage:      {coverage*100:.1f}%  <-- {'✅ Good' if abs(coverage-0.9)<0.05 else '⚠️ Needs Calibration'}")
    print(f"Avg Interval Width:   ${width:,.0f}")
    print(f"Avg House Price:      ${avg_price:,.0f}")
    print(f"Relative Uncertainty: {width/avg_price*100:.1f}% of price")

# Run the evaluation
evaluate_uncertainty(y_test)
