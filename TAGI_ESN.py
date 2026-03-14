# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import mean_squared_error

# Fix seed for reproducibility
np.random.seed(42)

# %% [markdown]
# # 1. Generate Two Distinct Tasks
def get_task_data(freq=1.0, n_samples=600, seq_length=15, noise=0.1):
    x = np.linspace(0, 10 * np.pi, n_samples)
    y_true = np.sin(freq * x)
    y_noisy = y_true + np.random.randn(n_samples) * noise
    
    X_seq, Y_seq, Y_true_seq = [], [], []
    for i in range(len(y_noisy) - seq_length):
        X_seq.append(y_noisy[i : i + seq_length])
        Y_seq.append(y_noisy[i + seq_length])
        Y_true_seq.append(y_true[i + seq_length])
        
    return np.array(X_seq)[..., np.newaxis], np.array(Y_seq), np.array(Y_true_seq)

seq_length = 20
noise_level = 0.1

# Task A: Slow Wave
X_A, Y_A, Y_true_A = get_task_data(freq=1.0, seq_length=seq_length, noise=noise_level)
# Task B: Fast Wave
X_B, Y_B, Y_true_B = get_task_data(freq=2.5, seq_length=seq_length, noise=noise_level)

# Normalization (Using Task A stats to simulate a continuous data stream)
y_mean, y_std = np.mean(Y_A), np.std(Y_A)
X_A_norm, Y_A_norm = (X_A - y_mean) / y_std, (Y_A - y_mean) / y_std
X_B_norm, Y_B_norm = (X_B - y_mean) / y_std, (Y_B - y_mean) / y_std

var_obs = (noise_level / y_std)**2

# Split into Train and Test for both tasks
split = 400
X_A_train, Y_A_train = X_A_norm[:split], Y_A_norm[:split]
X_A_test, Y_A_test, Y_A_test_true = X_A_norm[split:], Y_A_norm[split:], Y_true_A[split:]

X_B_train, Y_B_train = X_B_norm[:split], Y_B_norm[:split]
X_B_test, Y_B_test, Y_B_test_true = X_B_norm[split:], Y_B_norm[split:], Y_true_B[split:]

# %% [markdown]
# # 2. TAGI Core Functions (Identical to previous script)
def hardtanh_forward_tagi(Z_l):
    mu_z, var_z = Z_l[0], Z_l[1]
    epsilon = 1e-8
    std_z = np.sqrt(np.maximum(var_z, epsilon))
    a, b = (-1.0 - mu_z) / std_z, (1.0 - mu_z) / std_z
    phi_a, phi_b = norm.pdf(a), norm.pdf(b)
    Phi_a, Phi_b = norm.cdf(a), norm.cdf(b)

    mu_a = -Phi_a + (1.0 - Phi_b) + mu_z * (Phi_b - Phi_a) + std_z * (phi_a - phi_b)
    E_y2 = Phi_a + (1.0 - Phi_b) + (mu_z**2 + var_z) * (Phi_b - Phi_a) + 2 * mu_z * std_z * (phi_a - phi_b) + var_z * (a * phi_a - b * phi_b)
    var_a = np.maximum(E_y2 - mu_a**2, epsilon)
    
    return (mu_a, var_a, Phi_b - Phi_a)

def linear_forward(A_prev, theta_out):
    mu_a_prev, var_a_prev = A_prev
    mu_theta, var_theta = theta_out
    mu_w, mu_b = mu_theta[:, :-1], mu_theta[:, -1].reshape(-1, 1)
    var_w, var_b = var_theta[:, :-1], var_theta[:, -1].reshape(-1, 1)
    
    mu_z = mu_w @ mu_a_prev + mu_b
    var_z = var_w @ var_a_prev + var_w @ (mu_a_prev**2) + (mu_w**2) @ var_a_prev + var_b
    cov_z_theta = np.concatenate((np.einsum('ji,kj->kji', mu_a_prev, var_w), 
                                  np.repeat(var_b[:, :, np.newaxis], mu_a_prev.shape[1], axis=2)), axis=1)
    return (mu_z, var_z, cov_z_theta)

def update_readout_weights(Y_pred, Z_out, y_batch, theta_out, var_obs):
    mu_y, var_y = Y_pred[0], Y_pred[1] + var_obs
    mu_z, var_z, cov_z_theta = Z_out
    mu_theta, var_theta = theta_out
    
    K = var_z / (var_y + 1e-8)
    mu_z_y = mu_z + K * (y_batch - mu_y)
    var_z_y = np.maximum(1e-8, var_z * (1 - K))
    
    J_theta = cov_z_theta / (var_z + 1e-8)[:, np.newaxis, :] 
    delta_mu_theta = np.einsum('kib,kb->kib', J_theta, mu_z_y - mu_z)
    delta_var_theta = np.einsum('kib,kb->kib', J_theta**2, var_z_y - var_z)

    return (mu_theta + np.mean(delta_mu_theta, axis=2), 
            np.maximum(1e-8, var_theta + np.mean(delta_var_theta, axis=2)))

# %% [markdown]
# # 3. Setup ESN
n_res = 150 # Larger reservoir to hold capacity for multiple tasks
W_in = np.random.randn(n_res, 1) * 0.5
W_res = np.random.randn(n_res, n_res)
W_res = W_res * (0.9 / np.max(np.abs(np.linalg.eigvals(W_res)))) 
W_in_sq, W_res_sq = W_in**2, W_res**2

# Initialize TAGI Readout Weights ONCE
std_init = np.sqrt(2.0 / n_res)
theta_out = (np.concatenate((np.random.normal(0, std_init, (1, n_res)), np.zeros((1, 1))), axis=1), 
             np.concatenate((np.ones((1, n_res)) * (1.0 / n_res), np.ones((1, 1))), axis=1))

# Helper function to run a sequence through the ESN
def run_esn_step(X_batch, theta, update=False, y_batch=None):
    current_batch_size = X_batch.shape[0]
    mu_A, var_A = np.zeros((n_res, current_batch_size)), np.zeros((n_res, current_batch_size))
    
    for t in range(seq_length):
        mu_x = X_batch[:, t, :].T
        mu_Z = W_in @ mu_x + W_res @ mu_A
        var_Z = W_in_sq @ np.zeros_like(mu_x) + W_res_sq @ var_A
        mu_A, var_A, _ = hardtanh_forward_tagi((mu_Z, var_Z))
        
    Z_out = linear_forward((mu_A, var_A), theta)
    if update and y_batch is not None:
        return update_readout_weights(Z_out, Z_out, y_batch.reshape(1, -1), theta, var_obs)
    return Z_out

# %% [markdown]
# # 4. Sequential Continual Learning

batch_size = 10

print("--- Phase 1: Training on Task A (Slow Wave) ---")
for epoch in range(3):
    for i in range(0, len(X_A_train), batch_size):
        theta_out = run_esn_step(X_A_train[i:i+batch_size], theta_out, update=True, y_batch=Y_A_train[i:i+batch_size])

print("--- Phase 2: Training on Task B (Fast Wave) ---")
# WE DO NOT RESET THE WEIGHTS. We keep the exact same theta_out learned from Task A.
for epoch in range(3):
    for i in range(0, len(X_B_train), batch_size):
        theta_out = run_esn_step(X_B_train[i:i+batch_size], theta_out, update=True, y_batch=Y_B_train[i:i+batch_size])

# %% [markdown]
# # 5. Testing Both Tasks
print("Generating Forecasts for both tasks...")

Z_out_test_A = run_esn_step(X_A_test, theta_out)
y_pred_A = (Z_out_test_A[0].flatten() * y_std) + y_mean

Z_out_test_B = run_esn_step(X_B_test, theta_out)
y_pred_B = (Z_out_test_B[0].flatten() * y_std) + y_mean

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Task A Plot
time_axis_A = np.arange(len(Y_A_test))
ax1.plot(time_axis_A, Y_A_test_true, 'k--', label='True Slow Wave')
ax1.plot(time_axis_A, y_pred_A, 'b-', linewidth=2, label='TAGI Prediction AFTER learning Task B')
ax1.set_title('Task A Performance (Testing for Catastrophic Forgetting)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# Task B Plot
time_axis_B = np.arange(len(Y_B_test))
ax2.plot(time_axis_B, Y_B_test_true, 'k--', label='True Fast Wave')
ax2.plot(time_axis_B, y_pred_B, 'r-', linewidth=2, label='TAGI Prediction on Task B')
ax2.set_title('Task B Performance (The Newly Learned Task)')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

print(f"Final Task A MSE: {mean_squared_error(Y_A_test, y_pred_A):.4f}")
print(f"Final Task B MSE: {mean_squared_error(Y_B_test, y_pred_B):.4f}")