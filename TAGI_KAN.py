# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import math

# fixe seed
np.random.seed(42)

# %% [markdown]
# # Génération des données avec bruit Gaussien (Identique)

# %%
def get_data(n_samples : int = 1000, noise : float = 3, name = 'linear', test=False):
    if test:
        if name == 'linear':
            x = np.linspace(-5, 5, n_samples)
            y = x**3 + np.random.randn(len(x))*noise
            X_test = (x, y)
            y = x**3
            X_true = (x, y)
            return X_test, X_true
        
    if name == 'linear':
        x = np.linspace(-5, 5, n_samples)
        y = x**3 + np.random.randn(len(x))*noise
        return x, y

noise = 9
x, y = get_data(n_samples=1000, noise=0)
x_train, y_train = get_data(n_samples=1000, noise=noise)
x_val, y_val = get_data(n_samples=100, noise=noise)
test_data, true_data = get_data(n_samples=100, noise=noise, test=True)

x_test, y_test = test_data
x_true_data, y_true_data = true_data

# Normalisation
x_train_norm = (x_train - np.mean(x_train)) / np.std(x_train)
y_train_norm = (y_train - np.mean(y_train)) / np.std(y_train)
x_test_norm = (x_test - np.mean(x_train)) / np.std(x_train)
y_test_norm = (y_test - np.mean(y_train)) / np.std(y_train)
x_val_norm = (x_val - np.mean(x_train)) / np.std(x_train)
y_val_norm = (y_val - np.mean(y_train)) / np.std(y_train)
x_norm = (x - np.mean(x_train)) / np.std(x_train)
y_norm = (y - np.mean(y_train)) / np.std(y_train)

standardized_noise = noise / np.std(y_train)

# %%
def initialize_kan_params(layer_dims, K=4, seed=None):
    """
    Initializes parameters for a TAGI-KAN using exact Hermite moment matching.
    Ensures that the output of each layer maintains Var(Z) ≈ 1.
    """
    if seed is not None:
        np.random.seed(seed)
        
    parameters = {}
    L = len(layer_dims) - 1
    
    # Precompute k! for k = 0, ..., K-1
    # For K=4, this is [1, 1, 2, 6]
    k_fact = np.array([math.factorial(k) for k in range(K)])
    
    for l in range(L):
        n_in = layer_dims[l]
        n_out = layer_dims[l+1]
        
        # Calculate the variance budget for each polynomial degree 'k'
        # V_k = 1 / (n_in * K * k!)
        total_var_budget = 1.0 / (n_in * K * k_fact) # Shape: (K,)
        
        # Reshape for NumPy broadcasting: (1, 1, K)
        total_var_budget = total_var_budget.reshape(1, 1, K)
        
        # Split the budget: 2/5 for var_of_means, 2/5 for mean_of_var, 1/5 for bias
        var_of_means = total_var_budget * (2/5)
        mean_of_var = total_var_budget * (2/5)
        var_b_budget = total_var_budget * (1/5)
        
        # Initialize means with random normal noise scaled by the factorial budget
        mu_w = np.random.normal(0, np.sqrt(var_of_means), (n_out, n_in, K))
        
        # Initialize variances deterministically based on the factorial budget
        var_w = np.ones((n_out, n_in, K)) * mean_of_var
        
        # Initialize bias with mean 0 and small variance (1/5 of budget)
        # Sum across K basis functions for bias contribution
        var_b_total = np.sum(var_b_budget, axis=2)  # Shape: (1, 1)
        mu_b = np.zeros(n_out)
        var_b = np.ones(n_out) * var_b_total.flatten() / n_out
        
        parameters[f'theta_{l}'] = (mu_w, var_w)
        parameters[f'bias_{l}'] = (mu_b, var_b)
    
    return parameters

def hermite_moments(mu_a, var_a, K=4):
    """
    Computes exact expected values and Jacobians of Hermite polynomials 
    for a Gaussian variable X ~ N(mu_a, var_a).
    
    Returns:
    E_H: Expected values, shape (n_in, K, B)
    dE_H: Jacobian w.r.t mu_a, shape (n_in, K, B)
    """
    n_in, B = mu_a.shape
    
    E_H = np.zeros((n_in, K, B))
    dE_H = np.zeros((n_in, K, B))
    
    # He_0(x) = 1
    E_H[:, 0, :] = 1.0
    dE_H[:, 0, :] = 0.0
    
    if K == 1:
        return E_H, dE_H
    
    # He_1(x) = x
    E_H[:, 1, :] = mu_a
    dE_H[:, 1, :] = 1.0
    
    # Use recurrence relation: He_{k+1}(x) = x*He_k(x) - k*He_{k-1}(x)
    # For expected values: E[He_{k+1}] = mu_a*E[He_k] + var_a*dE[He_k]/dmu - k*E[He_{k-1}]
    for k in range(2, K):
        E_H[:, k, :] = mu_a * E_H[:, k-1, :] + var_a * dE_H[:, k-1, :] - (k-1) * E_H[:, k-2, :]
        dE_H[:, k, :] = E_H[:, k-1, :] + mu_a * dE_H[:, k-1, :] - (k-1) * dE_H[:, k-2, :]
    
    return E_H, dE_H

# %% [markdown]
# ### Forward Pass (KAN)

# %%
def kan_layer_forward(A_prev, theta_l, bias_l, K=4):
    """
    Forward pass for a single KAN layer using exact Gaussian moments.
    """
    mu_a_prev, var_a_prev = A_prev # Shapes: (n_in, B)
    mu_w, var_w = theta_l          # Shapes: (n_out, n_in, K)
    mu_b, var_b = bias_l           # Shapes: (n_out,)
    
    epsilon = 1e-8
    
    # 1. Compute exact basis moments
    E_H, dE_H = hermite_moments(mu_a_prev, var_a_prev, K) # Shapes: (n_in, K, B)
    
    # 2. Compute Output Mean (Summing across inputs i and basis k) + bias
    mu_z = np.einsum('oik,ikb->ob', mu_w, E_H) + mu_b[:, np.newaxis] # Shape: (n_out, B)
    
    # 3. Compute Output Variance using Delta Method for inputs + exact for weights + bias
    # Jacobian of output j w.r.t input i
    J_a = np.einsum('oik,ikb->oib', mu_w, dE_H) # Shape: (n_out, n_in, B)
    
    var_z_weights = np.einsum('oik,ikb->ob', var_w, E_H**2)
    var_z_inputs = np.einsum('oib,ib->ob', J_a**2, var_a_prev)
    var_z_bias = var_b[:, np.newaxis]
    var_z = np.maximum(var_z_weights + var_z_inputs + var_z_bias, epsilon) # Shape: (n_out, B)
    
    # 4. Compute covariances for the backward pass
    cov_z_w = np.einsum('oik,ikb->oikb', var_w, E_H) # Shape: (n_out, n_in, K, B)
    cov_z_a = J_a * var_a_prev[np.newaxis, :, :]     # Shape: (n_out, n_in, B)
    cov_z_b = var_b[:, np.newaxis]                   # Shape: (n_out, B)
    
    return (mu_z, var_z), cov_z_w, cov_z_a, cov_z_b

def kan_forward(X_batch, parameters, layer_dims, K=4):
    caches = []
    L = len(layer_dims) - 1
    
    A_prev = (X_batch, np.zeros_like(X_batch))
    
    for l in range(L):
        theta_l = parameters[f'theta_{l}']
        bias_l = parameters[f'bias_{l}']
        Z_l, cov_z_w, cov_z_a, cov_z_b = kan_layer_forward(A_prev, theta_l, bias_l, K)
        caches.append({'A_prev': A_prev, 'theta': theta_l, 'bias': bias_l, 'Z': Z_l, 'cov_w': cov_z_w, 'cov_a': cov_z_a, 'cov_b': cov_z_b})
        A_prev = Z_l # In KANs, the edge functions ACT as the activation
        
    return Z_l, caches

# %% [markdown]
# ### Modèle d'observation et Backward Pass (Inférence)

# %%
def obs_model(Z_out, var_v):
    mu_z, var_z = Z_out
    return (mu_z, var_z + var_v)

def log_likelihood(Y, y_batch):
    mu_y, var_y = Y
    epsilon = 1e-8
    var_y_stable = var_y + epsilon
    log_ll_samples = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(var_y_stable) - 0.5 * (y_batch - mu_y)**2 / var_y_stable
    return np.sum(log_ll_samples)

def update_output(Y, Z_out, y_batch):
    mu_y, var_y = Y
    mu_z, var_z = Z_out
    epsilon = 1e-8
    K = var_z / (var_y + epsilon)
    mu_z_y = mu_z + K * (y_batch - mu_y)
    var_z_y = np.maximum(epsilon, var_z * (1 - K))
    return (mu_z_y, var_z_y)

def kan_backward(Y, Z_out, y_batch, caches, parameters, layer_dims):
    """
    Backward inference pass. Elegantly simple due to the KAN structure!
    """
    updated_params = {}
    L = len(layer_dims) - 1
    
    Z_next_y = update_output(Y, Z_out, y_batch)
    epsilon = 1e-8
    
    for l in reversed(range(L)):
        cache = caches[l]
        mu_z, var_z = cache['Z']
        mu_z_y, var_z_y = Z_next_y
        mu_w, var_w = cache['theta']
        mu_b, var_b = cache['bias']
        
        delta_mu_z = mu_z_y - mu_z
        delta_var_z = var_z_y - var_z
        var_z_stable = var_z + epsilon
        
        # 1. Update Weights
        # cov_w shape: (n_out, n_in, K, B)
        J_w = cache['cov_w'] / var_z_stable[:, np.newaxis, np.newaxis, :] 
        
        delta_mu_w = np.mean(J_w * delta_mu_z[:, np.newaxis, np.newaxis, :], axis=3)
        delta_var_w = np.mean(J_w**2 * delta_var_z[:, np.newaxis, np.newaxis, :], axis=3)
        
        mu_w_y = mu_w + delta_mu_w
        var_w_y = np.maximum(epsilon, var_w + delta_var_w)
        updated_params[f'theta_{l}'] = (mu_w_y, var_w_y)
        
        # 2. Update Bias
        # cov_b shape: (n_out, B)
        J_b = cache['cov_b'] / var_z_stable
        
        delta_mu_b = np.mean(J_b * delta_mu_z, axis=1)
        delta_var_b = np.mean(J_b**2 * delta_var_z, axis=1)
        
        mu_b_y = mu_b + delta_mu_b
        var_b_y = np.maximum(epsilon, var_b + delta_var_b)
        updated_params[f'bias_{l}'] = (mu_b_y, var_b_y)
        
        # 3. Update Hidden States (to pass down to next layer)
        if l > 0:
            # cov_a shape: (n_out, n_in, B)
            J_a = cache['cov_a'] / var_z_stable[:, np.newaxis, :]
            
            # Sum the gradient-like updates coming from all output nodes
            delta_mu_a = np.einsum('oib,ob->ib', J_a, delta_mu_z)
            delta_var_a = np.einsum('oib,ob->ib', J_a**2, delta_var_z)
            
            mu_a, var_a = cache['A_prev']
            mu_a_y = mu_a + delta_mu_a
            var_a_y = np.maximum(epsilon, var_a + delta_var_a)
            
            Z_next_y = (mu_a_y, var_a_y) # Set up for previous layer
            
    return updated_params

# %% [markdown]
# ### Entraînement

# %%
batch_size = 10
K = 10  # Number of Hermite basis functions
layer_dims = [1, 100, 1] # Input feature is 1D
parameters = initialize_kan_params(layer_dims, K=K, seed=42)
best_params = parameters.copy()

n_epochs = 100
log_ll_train = []
log_ll_val = []
log_ref = -np.inf

x_train_batch = x_train_norm.reshape(1, -1)
y_train_batch = y_train_norm.reshape(1, -1)
x_val_batch = x_val_norm.reshape(1, -1)
y_val_batch = y_val_norm.reshape(1, -1)
x_test_batch = x_test_norm.reshape(1, -1)
y_test_batch = y_test_norm.reshape(1, -1)

n_samples_train = x_train_batch.shape[1]

for epoch in tqdm(range(n_epochs), desc="Training KAN Epochs"):
    epoch_log_likelihood_train = 0
    indices = np.random.permutation(n_samples_train)
    x_train_shuffled = x_train_batch[:, indices]
    y_train_shuffled = y_train_batch[:, indices]
    
    for i in range(0, n_samples_train, batch_size):
        end_idx = min(i + batch_size, n_samples_train)
        x_batch_i = x_train_shuffled[:, i:end_idx]
        y_batch_i = y_train_shuffled[:, i:end_idx]

        # Forward
        Z_out, caches = kan_forward(x_batch_i, parameters, layer_dims, K)
        Y = obs_model(Z_out, standardized_noise**2)
        batch_log_likelihood = log_likelihood(Y, y_batch_i)
        
        # Backward (Inference)
        updated_params = kan_backward(Y, Z_out, y_batch_i, caches, parameters, layer_dims)
        parameters.update(updated_params) 
        
        epoch_log_likelihood_train += batch_log_likelihood

    log_ll_train.append(epoch_log_likelihood_train / n_samples_train)

    # Validation
    total_log_likelihood_val = 0
    n_samples_val = x_val_batch.shape[1]
    for i in range(0, n_samples_val, batch_size):
         end_idx = min(i + batch_size, n_samples_val)
         x_val_batch_i = x_val_batch[:, i:end_idx]
         y_val_batch_i = y_val_batch[:, i:end_idx]

         Z_out_val, _ = kan_forward(x_val_batch_i, parameters, layer_dims, K)
         Y_val = obs_model(Z_out_val, standardized_noise**2)
         total_log_likelihood_val += log_likelihood(Y_val, y_val_batch_i)

    avg_log_ll_val = total_log_likelihood_val / n_samples_val
    log_ll_val.append(avg_log_ll_val)
    
    if avg_log_ll_val > log_ref:
        log_ref = avg_log_ll_val
        best_params = parameters.copy()

best_epoch_idx = np.argmax(log_ll_val)
print(f'Best validation log-likelihood ({log_ll_val[best_epoch_idx]:.4f}) at epoch: {best_epoch_idx}')
parameters = best_params 

# %% [markdown]
# ### Résultats et Plots

# %%
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 14})
plt.plot(log_ll_train, label='Training', color='steelblue', linewidth=2)
plt.plot(log_ll_val, label='Validation', color='grey', linewidth=2)
plt.axvline(best_epoch_idx, color='red', linestyle='--', label='Best Epoch')
plt.xlabel('Epochs')
plt.ylabel('Log-Likelihood')
plt.title('TAGI-KAN: Log-Likelihood')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()  
plt.show()

# Evaluation on Test Data
y_pred_list, var_pred_list = [], []
n_samples_test = x_test_batch.shape[1]

for i in range(0, n_samples_test, batch_size):
    end_idx = min(i + batch_size, n_samples_test)
    x_test_batch_i = x_test_batch[:, i:end_idx]
    
    Z_out_test, _ = kan_forward(x_test_batch_i, best_params, layer_dims, K)
    Y_test = obs_model(Z_out_test, standardized_noise**2)
    
    y_pred_list.append(Y_test[0])
    var_pred_list.append(Y_test[1])

y_pred = np.concatenate(y_pred_list, axis=1).flatten()
var_pred = np.concatenate(var_pred_list, axis=1).flatten()

# Rescale predictions
y_pred_orig = np.array(y_pred) * np.std(y_train) + np.mean(y_train)
var_pred_orig = np.array(var_pred) * np.std(y_train)**2

plt.figure(figsize=(8, 6))
plt.scatter(x_true_data, y_true_data, label='True Data', color='forestgreen', alpha=0.7, marker='x')
plt.plot(x_test, y_pred_orig, label='KAN Predicted Mean', color='darkorange', linewidth=2)
plt.fill_between(x_test, 
                 y_pred_orig - np.sqrt(var_pred_orig), 
                 y_pred_orig + np.sqrt(var_pred_orig), 
                 alpha=0.3, label='KAN Uncertainty (1 std)', color='darkorange')

plt.xlabel('x')
plt.ylabel('y')
plt.title('TAGI-KAN: True vs Predicted with Hermite Edges')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Metrics
print("\nTAGI-KAN Performance on True Data:")
print("{:<20} {:.2f}".format("Mean Squared Error:", mean_squared_error(y_pred_orig, y_true_data)))
print("{:<20} {:.2f}".format("Mean Absolute Error:", mean_absolute_error(y_pred_orig, y_true_data)))
print("{:<20} {:.2f}".format("R-squared:", r2_score(y_pred_orig, y_true_data)))