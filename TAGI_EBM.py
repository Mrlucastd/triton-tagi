# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

np.random.seed(42)

# ==========================================
# 1. Data Generation
# ==========================================
def get_data(n_samples=1000, noise=3, test=False):
    x = np.linspace(-5, 5, n_samples)
    y = x**3 + np.random.randn(len(x))*noise
    if test:
        y_true = x**3
        return (x, y), (x, y_true)
    return x, y

noise = 9
x_train, y_train = get_data(n_samples=1000, noise=noise)
test_data, true_data = get_data(n_samples=100, noise=noise, test=True)

x_test, y_test = test_data
x_true_data, y_true_data = true_data

# Normalization
x_mean, x_std = np.mean(x_train), np.std(x_train)
y_mean, y_std = np.mean(y_train), np.std(y_train)

x_train_norm = (x_train - x_mean) / x_std
y_train_norm = (y_train - y_mean) / y_std
x_test_norm = (x_test - x_mean) / x_std

# ==========================================
# 2. TAGI Core Functions (Exact Closed-Form)
# ==========================================
def initialize_params(layer_dims, seed=None):
    if seed is not None: np.random.seed(seed)
    parameters = {}
    L = len(layer_dims) - 1
    for l in range(L):
        mu_w = np.random.normal(0, np.sqrt(2/layer_dims[l]), (layer_dims[l+1], layer_dims[l]))
        var_w = np.ones((layer_dims[l+1], layer_dims[l])) * (1 / layer_dims[l])
        mu_b = np.zeros((layer_dims[l+1], 1))
        var_b = np.ones((layer_dims[l+1], 1)) * (1 / layer_dims[l])
        
        mu_theta = np.concatenate((mu_w, mu_b), axis=1)
        var_theta = np.concatenate((var_w, var_b), axis=1)
        parameters[f'theta_{l}'] = (mu_theta, var_theta)
    return parameters

def linear_forward(A_prev, theta_l, l):
    mu_a_prev, var_a_prev = A_prev[0], A_prev[1]
    mu_theta, var_theta = theta_l
    mu_w, mu_b = mu_theta[:, :-1], mu_theta[:, -1].reshape(-1, 1)
    var_w, var_b = var_theta[:, :-1], var_theta[:, -1].reshape(-1, 1)
    
    mu_z = mu_w @ mu_a_prev + mu_b
    var_z = var_w @ var_a_prev + var_w @ (mu_a_prev**2) + (mu_w**2) @ var_a_prev + var_b

    batch_size = mu_a_prev.shape[1]
    cov_z_w = np.einsum('ji,kj->kji', mu_a_prev, var_w) 
    cov_z_b = np.repeat(var_b[:, :, np.newaxis], batch_size, axis=2)
    cov_z_theta = np.concatenate((cov_z_w, cov_z_b), axis=1)

    return (mu_z, var_z, cov_z_theta)

def activation_forward(Z_l, activation_type='relu'):
    mu_z, var_z, _ = Z_l
    epsilon = 1e-8

    if activation_type == 'relu':
        std_z = np.sqrt(np.maximum(var_z, epsilon))
        alpha = mu_z / std_z
        
        cdf_alpha = norm.cdf(alpha)
        pdf_alpha = norm.pdf(alpha)
        
        mu_a = mu_z * cdf_alpha + std_z * pdf_alpha
        E_a2 = (mu_z**2 + var_z) * cdf_alpha + mu_z * std_z * pdf_alpha
        var_a = np.maximum(epsilon, E_a2 - mu_a**2)
        T = cdf_alpha
        
        return (mu_a, var_a, T)
    elif activation_type == 'linear':
        return (mu_z, var_z, np.ones_like(mu_z))

def model_forward(X_batch, parameters, layer_dims):
    caches = []
    L = len(layer_dims) - 1
    A_prev = (X_batch, np.zeros_like(X_batch))

    for l in range(L - 1):
        theta_l = parameters[f'theta_{l}']
        Z_l = linear_forward(A_prev, theta_l, l)
        A_l = activation_forward(Z_l, activation_type='relu')
        caches.append({'Z': Z_l, 'A_prev': A_prev, 'theta': theta_l, 'A': A_l})
        A_prev = A_l

    theta_out = parameters[f'theta_{L-1}']
    Z_out = linear_forward(A_prev, theta_out, L-1)
    A_out = activation_forward(Z_out, activation_type='linear')
    caches.append({'Z': Z_out, 'A_prev': A_prev, 'theta': theta_out, 'A': A_out})

    return Z_out, caches

def obs_model(Z_out, var_v):
    mu_z, var_z, _ = Z_out
    return (mu_z, var_z + var_v)

def update_output(Y, Z_out, y_batch):
    mu_y, var_y = Y
    mu_z, var_z, _ = Z_out
    epsilon = 1e-8
    var_y_stable = var_y + epsilon
    
    K = var_z / var_y_stable
    mu_z_y = mu_z + K * (y_batch - mu_y)
    var_z_y = np.maximum(epsilon, var_z * (1 - K))
    return (mu_z_y, var_z_y)

def update_parameters(Z_l, Z_l_y, theta_l):
    mu_z, var_z, cov_z_theta = Z_l
    mu_z_y, var_z_y = Z_l_y
    mu_theta, var_theta = theta_l

    epsilon = 1e-8 
    var_z_stable = var_z + epsilon
    J_theta = cov_z_theta / var_z_stable[:, np.newaxis, :] 

    delta_mu_z = mu_z_y - mu_z
    delta_var_z = var_z_y - var_z

    delta_mu_theta = np.einsum('kib,kb->kib', J_theta, delta_mu_z)
    delta_var_theta = np.einsum('kib,kb->kib', J_theta**2, delta_var_z)

    mu_theta_y = mu_theta + np.mean(delta_mu_theta, axis=2)
    var_theta_y = np.maximum(epsilon, var_theta + np.mean(delta_var_theta, axis=2))
    return (mu_theta_y, var_theta_y)

def update_hidden_state(Z_l, Z_l_plus_1, Z_l_plus_1_y, A_l, theta_l_plus_1):
    mu_z, var_z, _ = Z_l 
    mu_z_next, var_z_next, _ = Z_l_plus_1
    mu_z_next_y, var_z_next_y = Z_l_plus_1_y
    _, _, T = A_l

    mu_theta_next, _ = theta_l_plus_1
    mu_w_next = mu_theta_next[:, :-1]

    epsilon = 1e-8
    var_z_next_stable = var_z_next + epsilon
    cov_z_next_z = np.einsum('kj,jb,jb->kjb', mu_w_next, var_z, T)
    J_z = cov_z_next_z / var_z_next_stable[:, np.newaxis, :]

    delta_mu_z_next = mu_z_next_y - mu_z_next
    delta_var_z_next = var_z_next_y - var_z_next
    
    delta_mu_z = np.einsum('kji,ki->ji', J_z, delta_mu_z_next)
    delta_var_z = np.einsum('kji,ki->ji', J_z**2, delta_var_z_next)

    return (mu_z + delta_mu_z, np.maximum(epsilon, var_z + delta_var_z))

def model_backward(Y, Z_out, y_batch, caches, parameters, layer_dims):
    updated_params = {}
    L = len(layer_dims) - 1

    Z_out_y = update_output(Y, Z_out, y_batch)
    Z_next_y = Z_out_y

    cache_out = caches[L-1]
    updated_params[f'theta_{L-1}'] = update_parameters(cache_out['Z'], Z_out_y, cache_out['theta'])

    for l in reversed(range(L - 1)):
        cache_l = caches[l]
        cache_l_plus_1 = caches[l+1]
        
        Z_l = cache_l['Z']
        A_l = cache_l['A']
        theta_l = cache_l['theta']
        Z_l_plus_1 = cache_l_plus_1['Z']
        theta_l_plus_1 = parameters[f'theta_{l+1}']

        Z_l_y = update_hidden_state(Z_l, Z_l_plus_1, Z_next_y, A_l, theta_l_plus_1)
        updated_params[f'theta_{l}'] = update_parameters(Z_l, Z_l_y, theta_l)
        Z_next_y = Z_l_y
        
    return updated_params

# ==========================================
# 3. TAGI EBM Training (Contrastive Divergence)
# ==========================================
# Input is [x, y], Output is [Energy]
layer_dims = [2, 64, 64, 1] 
parameters = initialize_params(layer_dims, seed=42)

batch_size = 32
n_epochs = 100

# Energy targets
target_energy_real = 0.0
target_energy_fake = 5.0 # Margin M
obs_variance = 0.1 # Small variance for the energy observation

x_train_batch = x_train_norm.reshape(1, -1)
y_train_batch = y_train_norm.reshape(1, -1)
n_samples_train = x_train_batch.shape[1]

for epoch in tqdm(range(n_epochs), desc="Training TAGI EBM"):
    indices = np.random.permutation(n_samples_train)
    x_shuffled = x_train_batch[:, indices]
    y_shuffled = y_train_batch[:, indices]
    
    for i in range(0, n_samples_train, batch_size):
        end_idx = min(i + batch_size, n_samples_train)
        curr_batch_size = end_idx - i
        
        x_batch_i = x_shuffled[:, i:end_idx]
        y_batch_i = y_shuffled[:, i:end_idx]
        
        # --- POSITIVE PHASE (Push real energy down) ---
        X_pos = np.vstack((x_batch_i, y_batch_i)) # Shape: (2, B)
        E_pos_target = np.ones((1, curr_batch_size)) * target_energy_real
        
        Z_out_pos, caches_pos = model_forward(X_pos, parameters, layer_dims)
        Y_pos = obs_model(Z_out_pos, obs_variance)
        
        updated_params_pos = model_backward(Y_pos, Z_out_pos, E_pos_target, caches_pos, parameters, layer_dims)
        parameters.update(updated_params_pos) 
        
        # --- NEGATIVE PHASE (Push fake energy up) ---
        # Generate random fake y values uniformly across the normalized domain
        y_fake = np.random.uniform(-3, 3, size=(1, curr_batch_size))
        X_neg = np.vstack((x_batch_i, y_fake))
        E_neg_target = np.ones((1, curr_batch_size)) * target_energy_fake
        
        Z_out_neg, caches_neg = model_forward(X_neg, parameters, layer_dims)
        Y_neg = obs_model(Z_out_neg, obs_variance)
        
        updated_params_neg = model_backward(Y_neg, Z_out_neg, E_neg_target, caches_neg, parameters, layer_dims)
        parameters.update(updated_params_neg)

# ==========================================
# 4. Grid-Search Evaluation & Uncertainty
# ==========================================
print("\nEvaluating Energy Landscape...")
# Create a grid of possible y values to search over
y_grid_norm = np.linspace(-3, 3, 200).reshape(1, -1)
n_grid = y_grid_norm.shape[1]

y_pred_mean = []
y_pred_var = []

# For each test x, we evaluate the energy of all possible y's in the grid
for i in range(x_test_norm.shape[0]):
    x_val = x_test_norm[i]
    x_rep = np.repeat(x_val, n_grid).reshape(1, -1)
    
    # Input to the EBM: [x_i repeated 200 times, y_grid]
    X_eval = np.vstack((x_rep, y_grid_norm))
    
    # Forward pass to get Energy distributions for all y's
    Z_out, _ = model_forward(X_eval, parameters, layer_dims)
    
    # TAGI outputs for the Energy
    mu_E = Z_out[0].flatten()
    var_E = Z_out[1].flatten()
    
    # 1. Apply the closed-form moment matching for E[exp(-E)]
    log_unnorm = -mu_E + 0.5 * var_E
    
    # (Optional but recommended: shift by max to prevent np.exp() from overflowing to infinity)
    log_unnorm -= np.max(log_unnorm) 
    unnorm_probs = np.exp(log_unnorm)
    
    # 2. Calculate the partition function scalar Z (Ratio_int_Z)
    # We multiply by the grid step size (dy) to make it a true numerical integral
    dy = y_grid_norm[0, 1] - y_grid_norm[0, 0]
    Ratio_int_Z = np.sum(unnorm_probs) * dy
    
    # 3. Convert to a true normalized probability distribution
    probs = (unnorm_probs * dy) / Ratio_int_Z 
    
    # 4. Calculate expectation and variance over the y grid
    y_grid_flat = y_grid_norm.flatten()
    
    expected_y = np.sum(y_grid_flat * probs)
    variance_y = np.sum(((y_grid_flat - expected_y)**2) * probs)
    
    y_pred_mean.append(expected_y)
    y_pred_var.append(variance_y)

# Convert predictions back to original scale
y_pred_mean_orig = np.array(y_pred_mean) * y_std + y_mean
y_pred_var_orig = np.array(y_pred_var) * (y_std**2)
y_pred_std_orig = np.sqrt(y_pred_var_orig)

# ==========================================
# 5. Plotting
# ==========================================
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 14})

plt.scatter(x_true_data, y_true_data, label='True Data', color='forestgreen', alpha=0.7, marker='x')
plt.plot(x_test, y_pred_mean_orig, label='TAGI-EBM Expected Value', color='darkorange', linewidth=2)

plt.fill_between(x_test, 
                 y_pred_mean_orig - 2*y_pred_std_orig, 
                 y_pred_mean_orig + 2*y_pred_std_orig, 
                 alpha=0.2, label=r'Energy Implied Uncertainty ($\pm 2\sigma$)', color='darkorange')

plt.xlabel('x')
plt.ylabel('y')
plt.title('TAGI Energy-Based Model Inference')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()