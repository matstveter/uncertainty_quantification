# Databricks notebook source
# MAGIC %pip install torch torchvision
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

housing = fetch_california_housing()
X, y = housing.data, housing.target

# 1. Split: Train (60%) vs Temp (40%)
X_train_raw, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

# 2. Split Temp: Val (20%) vs Test (20%)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.log(torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.log(torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.log(torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# COMMAND ----------

class HeteroscedasticHousing(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(64, 1)    # Price prediction
        self.log_var_head = nn.Linear(64, 1) # Predicts Uncertainty

    def forward(self, x):
        features = self.feature_extractor(x)
        mu = self.mean_head(features)
        log_var = self.log_var_head(features)

        return mu, log_var
    
def heteroscedastic_loss(true_y, pred_mu, pred_log_var):
    precision = torch.exp(-pred_log_var)
    loss = 0.5 * torch.mean(pred_log_var + (true_y - pred_mu)**2 * precision)

    return loss

class ModelWithTemperature(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model 
        self.temperature_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mu, log_var = self.base_model(x) 
        return mu, log_var + self.temperature_bias

    def set_temperature(self, valid_loader):
        self.to(next(self.base_model.parameters()).device)
        nll_criterion = heteroscedastic_loss
        optimizer = optim.LBFGS([self.temperature_bias], lr=0.01, max_iter=50)

        # Pre-compute
        valid_logits_list = []
        valid_labels_list = []
        self.base_model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                mu, log_var = self.base_model(x_batch)
                valid_logits_list.append((mu, log_var))
                valid_labels_list.append(y_batch)

        def closure():
            optimizer.zero_grad()
            loss = 0
            for (mu, log_var), y in zip(valid_logits_list, valid_labels_list):
                calibrated_log_var = log_var + self.temperature_bias
                loss += nll_criterion(y, mu, calibrated_log_var)
            loss.backward()
            return loss

        optimizer.step(closure)
        final_T = torch.sqrt(torch.exp(self.temperature_bias)).item()
        print(f"Optimal Temperature (T): {final_T:.4f}")

# COMMAND ----------

model = HeteroscedasticHousing()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# COMMAND ----------

epochs = 2000
print(f"\nStarting Training (1000 Epochs)...")
print(f"{'Epoch':<10} | {'Loss':<12} | {'Avg Sigma':<15}")
print("-" * 45)

loss_history = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    pred_mu, pred_log_var = model(X_train)
    loss = heteroscedastic_loss(y_train, pred_mu, pred_log_var)

    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    
    if epoch % 100 == 0:
        avg_sigma = torch.exp(0.5 * pred_log_var).mean().item()
        print(f"{epoch:<10} | {loss.item():.4f}       | {avg_sigma:.4f}")

# COMMAND ----------

import numpy as np
import torch

def check_calibration(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        mu, log_var = model(X_test)
        sigma = torch.exp(0.5 * log_var)
    
    y_true = np.exp(y_test.numpy()) 
    mu = mu.numpy()
    sigma = sigma.numpy()
    
    z_score = 1.96
    lower_log = mu - (z_score * sigma)
    upper_log = mu + (z_score * sigma)
    
    lower_bound = np.exp(lower_log)
    upper_bound = np.exp(upper_log)
    
    is_inside = (y_true >= lower_bound) & (y_true <= upper_bound)
    coverage = np.mean(is_inside)
    
    print(f"Expected Coverage: 95.0%")
    print(f"Actual Coverage:   {coverage * 100:.2f}%")
    
    if coverage < 0.90:
        print("Diagnosis: OVER-CONFIDENT (Intervals are too narrow)")
    elif coverage > 0.99:
        print("Diagnosis: UNDER-CONFIDENT (Intervals are too wide)")
    else:
        print("Diagnosis: WELL CALIBRATED (Good job!)")
        
    return coverage

check_calibration(model, X_val, y_val)

# COMMAND ----------

from torch.utils.data import TensorDataset, DataLoader

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

scaled_model = ModelWithTemperature(model)
scaled_model.set_temperature(val_loader)

check_calibration(scaled_model, X_val, y_val)

# COMMAND ----------

import torch
import numpy as np
import matplotlib.pyplot as plt

print("--- FINAL AUDIT: Test Set Performance (Log-Transformed Data) ---")

criterion = heteroscedastic_loss
scaled_model.eval()
model.eval()

with torch.no_grad():
    # Raw Model NLL
    mu_log_raw, lv_log_raw = model(X_test)
    nll_raw = criterion(y_test, mu_log_raw, lv_log_raw).item()
    
    # Calibrated Model NLL
    mu_log_cal, lv_log_cal = scaled_model(X_test)
    nll_cal = criterion(y_test, mu_log_cal, lv_log_cal).item()

sigma_log_cal = torch.exp(0.5 * lv_log_cal)
low_log  = mu_log_cal - 1.96 * sigma_log_cal
high_log = mu_log_cal + 1.96 * sigma_log_cal

# Check if log-target falls inside log-bounds
covered_indices = (y_test.flatten() >= low_log.flatten()) & (y_test.flatten() <= high_log.flatten())
test_coverage = covered_indices.float().mean().item()

print(f"\nNegative Log Likelihood (NLL):")
print(f"Original:   {nll_raw:.4f}")
print(f"Calibrated: {nll_cal:.4f}")
print(f"Improvement: {nll_raw - nll_cal:.4f}")
print(f"Test Coverage: {test_coverage*100:.2f}% (Target: 95%)")

# We must exponentiate to visualize the "Trumpet" or "Spike" in real terms
y_true_real = torch.exp(y_test).numpy().flatten()
y_pred_real = torch.exp(mu_log_cal).numpy().flatten()
y_lower_real = torch.exp(low_log).numpy().flatten()
y_upper_real = torch.exp(high_log).numpy().flatten()

error_low = y_pred_real - y_lower_real
error_high = y_upper_real - y_pred_real
y_err_asym = [error_low, error_high]

# Sort by True Price for clean plotting
sort_idx = np.argsort(y_true_real)
y_true_sorted = y_true_real[sort_idx]
y_pred_sorted = y_pred_real[sort_idx]

# Sort the errors using the same indices
y_err_sorted = [error_low[sort_idx], error_high[sort_idx]]

# Plot
plt.figure(figsize=(12, 6))
subset = 100 # Plot first 100 samples for clarity

plt.plot(range(subset), y_true_sorted[:subset], 'k--', label='True Price', alpha=0.8)

# Note: yerr accepts shape (2, N) for asymmetric errors
plt.errorbar(range(subset), y_pred_sorted[:subset], 
             yerr=[y_err_sorted[0][:subset], y_err_sorted[1][:subset]], 
             fmt='o', ecolor='green', color='blue', alpha=0.5, 
             label='Calibrated Prediction (95% CI)')

plt.title(f"Final Test Set Result (Converted back to Real Values)\nCoverage: {test_coverage*100:.2f}%")
plt.xlabel("Test Samples (Sorted by Real Price)")
plt.ylabel("Price ($100k)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# COMMAND ----------

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def audit_sharpness(y_true, y_pred, y_lower, y_upper):
    """
    Audits the "sharpness" and "adaptivity" of uncertainty intervals.
    """
    # 1. Calculate Widths
    widths = y_upper - y_lower
    
    # 2. Metric: MPIW (Mean Prediction Interval Width)
    mpiw = np.mean(widths)
    target_range = np.max(y_true) - np.min(y_true)
    mpiw_vs_range = (mpiw / target_range) * 100  # As % of data range
    
    # 3. Metric: Width Variability (Check for "Laziness")
    # If std is 0, the model is just adding a constant buffer (Homoscedastic)
    width_std = np.std(widths)
    
    # 4. Metric: Error-Uncertainty Correlation (The "Trust" correlation)
    # do we get wider when we have higher error?
    abs_error = np.abs(y_true - y_pred)
    # We use Spearman because the relationship is likely monotonic but not linear
    corr, _ = stats.spearmanr(widths, abs_error)
    
    print("--- SHARPNESS AUDIT ---")
    print(f"Mean Width (MPIW):       {mpiw:.4f}")
    print(f"MPIW as % of Range:      {mpiw_vs_range:.2f}% (Lower is better)")
    print(f"Width Std Dev:           {width_std:.4f} (Higher implies adaptivity)")
    print(f"Error-Width Correlation: {corr:.4f} (Should be > 0)")
    
    # Quick Visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram of widths (Check for laziness)
    ax[0].hist(widths, bins=30, alpha=0.7, color='teal')
    ax[0].set_title("Distribution of Interval Widths")
    ax[0].set_xlabel("Width")
    ax[0].set_ylabel("Frequency")
    
    # Error vs Uncertainty
    ax[1].scatter(widths, abs_error, alpha=0.3, s=10)
    ax[1].set_title(f"Error vs Uncertainty (Corr: {corr:.2f})")
    ax[1].set_xlabel("Predicted Interval Width")
    ax[1].set_ylabel("Actual Absolute Error")
    
    plt.show()

# 1. Get Model Outputs (in Log Space)
scaled_model.eval()
with torch.no_grad():
    mu_log, lv_log = scaled_model(X_test)
    sigma_log = torch.exp(0.5 * lv_log)

# 2. Calculate Bounds in Log Space FIRST
# We apply the +/- 1.96 sigma HERE, in the Gaussian domain where the model was trained
z_score = 1.96
low_log  = mu_log - (z_score * sigma_log)
high_log = mu_log + (z_score * sigma_log)

# 3. Transform EVERYTHING back to Real Values ($$$)
# We want to audit "Dollars", not "Log-Dollars"
y_true_real = torch.exp(y_test).numpy().flatten()
y_pred_real = torch.exp(mu_log).numpy().flatten()
y_lower_real = torch.exp(low_log).numpy().flatten()
y_upper_real = torch.exp(high_log).numpy().flatten()

# 4. Run the Audit
audit_sharpness(y_true_real, y_pred_real, y_lower_real, y_upper_real)

# COMMAND ----------

import numpy as np
import torch

scaled_model.eval()
with torch.no_grad():
    # A. Validation Set (Calibration Data)
    # y_val is ALREADY log-transformed (Log-Dollars)
    mu_val_log, log_var_val = scaled_model(X_val)
    sigma_val_log = torch.exp(0.5 * log_var_val)
    
    # B. Test Set (Target Data)
    # y_test is ALREADY log-transformed (Log-Dollars)
    mu_test_log, log_var_test = scaled_model(X_test)
    sigma_test_log = torch.exp(0.5 * log_var_test)

# Convert to Numpy
y_cal_log_true = y_val.numpy().flatten() 
pred_cal_log   = mu_val_log.numpy().flatten()
sigma_cal_log  = sigma_val_log.numpy().flatten()

pred_test_log  = mu_test_log.numpy().flatten()
sigma_test_log = sigma_test_log.numpy().flatten()


# --- 2. RUN CONFORMAL PREDICTION (LOG-SPACE) ---

def apply_conformal_prediction(y_cal, pred_cal, sigma_cal, pred_test, sigma_test, alpha=0.05):
    # 1. Calculate Non-Conformity Scores
    # We are comparing Log-Truth to Log-Prediction. This is correct.
    scores = np.abs(y_cal - pred_cal) / sigma_cal
    
    # 2. Find q_hat (95th percentile)
    q_level = np.ceil((len(y_cal) + 1) * (1 - alpha)) / len(y_cal)
    q_level = min(1.0, q_level)
    q_hat = np.quantile(scores, q_level, method='higher')
    
    print(f"--- CONFORMAL PREDICTION ---")
    print(f"Standard Gaussian Multiplier: 1.96")
    print(f"Conformal Multiplier (q_hat): {q_hat:.4f}")
    
    # 3. Apply q_hat to Test Data (Still in Log Space)
    cp_lower_log = pred_test - (q_hat * sigma_test)
    cp_upper_log = pred_test + (q_hat * sigma_test)
    
    return cp_lower_log, cp_upper_log

# Get the calibrated bounds
cp_lower_log, cp_upper_log = apply_conformal_prediction(
    y_cal_log_true, pred_cal_log, sigma_cal_log, 
    pred_test_log, sigma_test_log
)


y_true_real = np.exp(y_test.numpy().flatten()) 
y_pred_real = np.exp(pred_test_log)

# Bounds (Log -> Real)
cp_lower_real = np.exp(cp_lower_log)
cp_upper_real = np.exp(cp_upper_log)

audit_sharpness(y_true_real, y_pred_real, cp_lower_real, cp_upper_real)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

# We use the back-transformed "Real Dollar" values calculated in the previous step
y_true = y_true_real      # The actual prices
y_pred = y_pred_real      # The exponentiated model mean
cp_lower = cp_lower_real  # The exponentiated lower bound
cp_upper = cp_upper_real  # The exponentiated upper bound

def plot_sorted_intervals(y_true, y_pred, y_lower, y_upper, num_points=100):
    """
    Plots a random sample of predictions sorted by the ground truth value.
    """
    if len(y_true) > num_points:
        indices = np.random.choice(len(y_true), num_points, replace=False)
    else:
        indices = np.arange(len(y_true))
        
    y_t_sample = y_true[indices]
    y_p_sample = y_pred[indices]
    y_l_sample = y_lower[indices]
    y_u_sample = y_upper[indices]

    # 2. Sort by Ground Truth (y_true)
    sort_idx = np.argsort(y_t_sample)
    y_t_sorted = y_t_sample[sort_idx]
    y_p_sorted = y_p_sample[sort_idx]
    y_l_sorted = y_l_sample[sort_idx]
    y_u_sorted = y_u_sample[sort_idx]

    # 3. Plot
    plt.figure(figsize=(12, 6))
    
    # Plot the Interval (Region)
    plt.fill_between(range(len(y_t_sorted)), y_l_sorted, y_u_sorted, 
                     color='gray', alpha=0.3, label='95% Confidence Interval')
    
    # Plot the Predictions (Model)
    plt.plot(range(len(y_t_sorted)), y_p_sorted, color='blue', 
             linewidth=1.5, alpha=0.8, label='Prediction')
    
    # Plot the Ground Truth (Reality)
    plt.scatter(range(len(y_t_sorted)), y_t_sorted, color='red', 
                s=15, alpha=0.7, label='Ground Truth')
    
    plt.title("FINAL RESULT: Log-Transformed & Conformalized\n(Note the asymmetric bounds tracking the trend)")
    plt.legend()
    plt.xlabel("Sample Index (Sorted by Target Value)")
    plt.ylabel("Price ($100k)")
    plt.show()

# --- 2. EXECUTION ---
plot_sorted_intervals(y_true, y_pred, cp_lower, cp_upper, num_points=200)

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew

# Ensure y_true is a 1D numpy array
y_target = y_test.numpy().flatten()

# 1. Calculate Skewness
# 0 = Normal, > 1 = Highly Skewed (Long Tail)
skew_val = skew(y_target)

print(f"--- TARGET DISTRIBUTION AUDIT ---")
print(f"Skewness Score: {skew_val:.4f}")
if skew_val > 1.0:
    print("Diagnosis: HIGHLY SKEWED (Long Tail detected)")
    print("Recommendation: Log-Transform your targets.")
elif skew_val < -1.0:
    print("Diagnosis: HIGHLY SKEWED (Negative)")
else:
    print("Diagnosis: SYMMETRIC (Normal-ish)")

# 2. Plot Histogram
plt.figure(figsize=(10, 5))
sns.histplot(y_target, kde=True, bins=30, color='purple', alpha=0.6)
plt.title(f"Distribution of Target Values (Skew: {skew_val:.2f})")
plt.xlabel("Target Value")
plt.ylabel("Frequency")
plt.axvline(x=np.mean(y_target), color='red', linestyle='--', label=f'Mean: {np.mean(y_target):.2f}')
plt.axvline(x=np.median(y_target), color='blue', linestyle='--', label=f'Median: {np.median(y_target):.2f}')
plt.legend()
plt.show()

# COMMAND ----------

import numpy as np

# 1. Pick a random test sample
idx = np.random.randint(0, len(y_true_real))

# 2. Extract the values (already in Real Dollars from previous step)
true_price  = y_true_real[idx]
pred_price  = y_pred_real[idx]
lower_bound = cp_lower_real[idx]
upper_bound = cp_upper_real[idx]

# 3. Display the result
print(f"--- DUMMY PREDICTION (Sample #{idx}) ---")
print(f"True Value:       ${true_price * 100_000:,.2f}")
print(f"Model Prediction: ${pred_price * 100_000:,.2f}")
print(f"95% Safe Range:   [${lower_bound * 100_000:,.2f}  --  ${upper_bound * 100_000:,.2f}]")

is_inside = (true_price >= lower_bound) and (true_price <= upper_bound)
width_dollars = (upper_bound - lower_bound) * 100_000

print(f"\nVerdict:          {'SUCCESS' if is_inside else 'FAILURE'}")
print(f"Uncertainty Width: ${width_dollars:,.2f}")

if is_inside:
    print("The true price is within the model's predicted interval.")
else:
    print("The model failed to capture the true price.")

# COMMAND ----------


