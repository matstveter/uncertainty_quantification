# Databricks notebook source
# MAGIC %pip install torch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer, VectorAssembler
from pyspark.sql.functions import col
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# COMMAND ----------

# Define the path to the built-in dataset
file_path = "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv"

# Read file
df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(file_path)

# There are 20 rows with 0 in either x, y or z, so we need to remove those, it is few in relation to the larger dataset
df_clean = df.filter((col("x") > 0) & (col("y") > 0) & (col("z") > 0) )

# higly correlated with carat
df_final = df_clean.drop("x", "y", "z")

categoical_columns = ['cut', 'color', 'clarity']
number_columsn = ['carat', 'depth', 'table']
prediction_columns = ['price']

ordinal_sql_query = """
SELECT *,
    -- Cut Score Logic
    CASE 
        WHEN cut = 'Fair' THEN 1 
        WHEN cut = 'Good' THEN 2 
        WHEN cut = 'Very Good' THEN 3 
        WHEN cut = 'Premium' THEN 4 
        WHEN cut = 'Ideal' THEN 5 
        ELSE 0 
    END as cut_score,

    -- Color Score Logic
    CASE 
        WHEN color = 'J' THEN 1 
        WHEN color = 'I' THEN 2 
        WHEN color = 'H' THEN 3 
        WHEN color = 'G' THEN 4 
        WHEN color = 'F' THEN 5 
        WHEN color = 'E' THEN 6 
        WHEN color = 'D' THEN 7 
        ELSE 0 
    END as color_score,

    -- Clarity Score Logic
    CASE 
        WHEN clarity = 'I1' THEN 1 
        WHEN clarity = 'SI2' THEN 2 
        WHEN clarity = 'SI1' THEN 3 
        WHEN clarity = 'VS2' THEN 4 
        WHEN clarity = 'VS1' THEN 5 
        WHEN clarity = 'VVS2' THEN 6 
        WHEN clarity = 'VVS1' THEN 7 
        WHEN clarity = 'IF' THEN 8 
        ELSE 0 
    END as clarity_score

FROM __THIS__
"""
ordinal_stage = SQLTransformer(statement=ordinal_sql_query)

feature_columns = ['carat', 'depth', 'table', 'cut_score', 'color_score', 'clarity_score']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

pipeline = Pipeline(stages=[ordinal_stage, assembler])

model_df = pipeline.fit(df_final).transform(df_final)

# COMMAND ----------

final_data = model_df.select('features', 'price').toPandas()

# Extract X and y
# Spark's "features" column is a DenseVector, we need to stack them
X_raw = np.array(final_data['features'].tolist())
y_raw = final_data['price'].values.astype(np.float32)

# Log-transform Price (Crucial for regression on money)
y_raw = np.log1p(y_raw)

# Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to Tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

print(f"Ready for Bayes: {X_train_tensor.shape}")

# COMMAND ----------

class VariationalLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # Initialize Weights: Mean (mu) and Log-Variance (rho)
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features).uniform_(-0.1, 0.1))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features).uniform_(-3, -3))

        # Initialize Bias: Mean (mu) and Log-Variance (rho)
        self.b_mu = nn.Parameter(torch.zeros(output_features).uniform_(-0.1, 0.1))
        self.b_rho = nn.Parameter(torch.zeros(output_features).uniform_(-3, -3))
    
    def forward(self, x, sample=False):
        if not sample:
            return F.linear(x, self.w_mu, self.b_mu)
        
        # Convert rho to positive values
        w_sigma = torch.log1p(torch.exp(self.w_rho))
        b_sigma = torch.log1p(torch.exp(self.b_rho))

        # Sample noise
        w_eps = torch.randn_like(w_sigma)
        b_eps = torch.randn_like(b_sigma)

        # Construct the sampled weights
        w_sampled = self.w_mu + (w_sigma * w_eps)
        b_sampled = self.b_mu + (b_sigma * b_eps)

        return F.linear(x, w_sampled, b_sampled)

    def kl_divergence(self):
        
        w_sigma = torch.log1p(torch.exp(self.w_rho))
        b_sigma = torch.log1p(torch.exp(self.b_rho))

        return 0.5 * (self.w_mu**2 + w_sigma**2 - 2*torch.log(w_sigma) - 1).sum() + 0.5 * (self.b_mu**2 + b_sigma**2 - 2*torch.log(b_sigma) - 1).sum()
    
class BayesianDiamondModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.l1 = VariationalLinear(input_dim, 64)
        self.l2 = VariationalLinear(64, 32)
        self.l3 = VariationalLinear(32, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x, sample=False):
        x = self.activation(self.l1(x, sample))
        x = self.activation(self.l2(x, sample))
        x = self.l3(x, sample)
        return x
    
    def total_kl(self):
        return self.l1.kl_divergence() + self.l2.kl_divergence() + self.l3.kl_divergence()

# COMMAND ----------

model = BayesianDiamondModel(input_dim=X_train_tensor.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Hyperparameters
EPOCHS = 2000
BATCH_SIZE = 1024
# KL Weighting: This is crucial! It balances the "Data fit" vs "Prior belief"
# A common heuristic is 1 / num_training_samples
kl_weight = 1.0 / len(X_train_tensor)

loss_history = []

print("Starting Variational Inference Training...")

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    # 1. Forward Pass (WITH Sampling)
    # We sample weights *every* time, so the network never sees the exact same model twice
    preds = model(X_train_tensor, sample=True)
    
    # 2. Calculate Loss Components
    mse = F.mse_loss(preds, y_train_tensor, reduction='sum')
    kl = model.total_kl()
    
    # 3. ELBO Loss (Evidence Lower Bound)
    loss = mse + (kl * kl_weight)
    
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f} (MSE: {mse.item()/len(X_train_tensor):.4f})")

plt.plot(loss_history)
plt.title("Bayesian Training Curve (ELBO)")
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import torch

# --- VISUALIZATION: The Spaghetti Plot ---

# 1. Pick 20 random diamonds from the TEST set
# We want to see how the model "thinks" about data it hasn't seen
idxs = np.random.choice(len(X_test_tensor), 20, replace=False)
sample_X = X_test_tensor[idxs]
sample_y = y_test_tensor[idxs]

# 2. Monte Carlo Sampling
# Predict on the SAME 20 diamonds 100 times.
# Because the weights are distributions, each prediction will be slightly different.
print("Running Monte Carlo Inference...")
predictions = []
model.eval() # Technically we use the same forward pass logic, but good practice
with torch.no_grad():
    for _ in range(100):
        # sample=True is CRITICAL here
        predictions.append(model(sample_X, sample=True).numpy())

# Shape: (100 samples, 20 diamonds)
predictions = np.array(predictions).squeeze()

# 3. Calculate Stats (in Log Space first)
mean_log_preds = predictions.mean(axis=0)
std_log_preds = predictions.std(axis=0)

# 4. Convert back to Real Dollars
# We used log1p, so we use expm1 to reverse it
pred_price = np.expm1(mean_log_preds)
true_price = np.expm1(sample_y.numpy().flatten())

# Approximate 95% Interval (+/- 2 Standard Deviations)
# We calculate bounds in log space, then convert, to preserve the distribution shape
lower_bound = np.expm1(mean_log_preds - 2 * std_log_preds)
upper_bound = np.expm1(mean_log_preds + 2 * std_log_preds)

# 5. Plotting
plt.figure(figsize=(14, 7))
x_axis = np.arange(20)

# Plot the Truth (Red Dots)
plt.scatter(x_axis, true_price, color='red', s=60, label='True Price', zorder=10)

# Plot the Prediction (Blue Dots)
plt.scatter(x_axis, pred_price, color='blue', s=40, label='Mean Prediction', zorder=9)

# Plot the Uncertainty (Error Bars)
# This represents Epistemic Uncertainty (Model Knowledge)
plt.errorbar(x_axis, pred_price, 
             yerr=[pred_price - lower_bound, upper_bound - pred_price], 
             fmt='none', color='blue', alpha=0.4, capsize=5, label='95% Bayesian CI')

plt.title("Bayesian Uncertainty: Do we trust our predictions?")
plt.ylabel("Price ($)")
plt.xlabel("Diamond ID (Random Selection)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# --- DIAGNOSTIC REPORT ---
print(f"--- DIAGNOSTICS ---")
avg_std = np.mean(std_log_preds)
print(f"Average Model Variance (Log Scale): {avg_std:.4f}")
if avg_std < 0.01:
    print("WARNING: Uncertainty is collapsed. The model is acting like a standard NN (ignoring variance).")
    print("   Fix: Increase KL_WEIGHT or simplify the model.")
elif avg_std > 1.0:
    print("WARNING: Uncertainty is exploding. The model learned nothing.")
    print("   Fix: Decrease LEARNING RATE or KL_WEIGHT.")
else:
    print("SUCCESS: The model has learned a healthy balance of confidence and caution.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Trying with another model trained for longer

# COMMAND ----------

model2 = BayesianDiamondModel(input_dim=X_train_tensor.shape[1])
optimizer = optim.Adam(model2.parameters(), lr=0.005)

# Hyperparameters
EPOCHS = 2500
BATCH_SIZE = 1024
# KL Weighting: This is crucial! It balances the "Data fit" vs "Prior belief"
# A common heuristic is 1 / num_training_samples
kl_weight = 1.0 / len(X_train_tensor)

loss_history = []

print("Starting Variational Inference Training...")

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    # Forward Pass (WITH Sampling)
    preds = model2(X_train_tensor, sample=True)
    
    # Calculate Loss Components
    mse = F.mse_loss(preds, y_train_tensor, reduction='sum')
    kl = model2.total_kl()
    
    # ELBO Loss (Evidence Lower Bound)
    loss = mse + (kl * kl_weight)
    
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f} (MSE: {mse.item()/len(X_train_tensor):.4f})")

plt.plot(loss_history)
plt.title("Bayesian Training Curve (ELBO)")
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import torch

# Pick 20 random diamonds to predict
idxs = np.random.choice(len(X_test_tensor), 20, replace=False)
sample_X = X_test_tensor[idxs]
sample_y = y_test_tensor[idxs]

# 2. Monte Carlo Sampling
# Predict on the SAME 20 diamonds 100 times.
# Because the weights are distributions, each prediction will be slightly different.
print("Running Monte Carlo Inference...")
predictions = []
model2.eval() # Technically we use the same forward pass logic, but good practice
with torch.no_grad():
    for _ in range(100):
        # sample=True is CRITICAL here
        predictions.append(model2(sample_X, sample=True).numpy())

# Shape: (100 samples, 20 diamonds)
predictions = np.array(predictions).squeeze()

# Calculate Stats (in Log Space first)
mean_log_preds = predictions.mean(axis=0)
std_log_preds = predictions.std(axis=0)

# Convert back to dollars
pred_price = np.expm1(mean_log_preds)
true_price = np.expm1(sample_y.numpy().flatten())

# Approximate 95% Interval (+/- 2 Standard Deviations)
# Calculate bounds in log space, then convert, to preserve the distribution shape
lower_bound = np.expm1(mean_log_preds - 2 * std_log_preds)
upper_bound = np.expm1(mean_log_preds + 2 * std_log_preds)

plt.figure(figsize=(14, 7))
x_axis = np.arange(20)

# Plot the Truth (Red Dots)
plt.scatter(x_axis, true_price, color='red', s=60, label='True Price', zorder=10)

# Plot the Prediction (Blue Dots)
plt.scatter(x_axis, pred_price, color='blue', s=40, label='Mean Prediction', zorder=9)

# Plot the Uncertainty (Error Bars)
# This represents Epistemic Uncertainty: model knowledge
plt.errorbar(x_axis, pred_price, 
             yerr=[pred_price - lower_bound, upper_bound - pred_price], 
             fmt='none', color='blue', alpha=0.4, capsize=5, label='95% Bayesian CI')

plt.title("Bayesian Uncertainty: Do we trust our predictions?")
plt.ylabel("Price ($)")
plt.xlabel("Diamond ID (Random Selection)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# --- DIAGNOSTIC REPORT ---
print(f"--- DIAGNOSTICS ---")
avg_std = np.mean(std_log_preds)
print(f"Average Model2model2 Variance (Log Scale): {avg_std:.4f}")
if avg_std < 0.01:
    print("WARNING: Uncertainty is collapsed. The model2 is acting like a standard NN (ignoring variance).")
    print("   Fix: Increase KL_WEIGHT or simplify the model2.")
elif avg_std > 1.0:
    print("WARNING: Uncertainty is exploding. The model2 learned nothing.")
    print("   Fix: Decrease LEARNING RATE or KL_WEIGHT.")
else:
    print("SUCCESS: The model2 has learned a healthy balance of confidence and caution.")

# COMMAND ----------

print(f"--- Example: Diamond #0 ---")
print(f"True Price:       ${true_price[0]:,.2f}")
print(f"Model Prediction: ${pred_price[0]:,.2f}")
print(f"Uncertainty Range: ${lower_bound[0]:,.2f} to ${upper_bound[0]:,.2f}")
print(f"Range Width:      ${(upper_bound[0] - lower_bound[0]):,.2f}")
