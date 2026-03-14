# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Fix seed for reproducibility
np.random.seed(42)

# %% [markdown]
# # 1. Generation of the Sine Wave Dataset

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

noise_level = 9
x_train, y_train = get_data(n_samples=800, noise=noise_level)
x_val, y_val = get_data(n_samples=200, noise=noise_level)
(x_test, y_test), (x_true, y_true) = get_data(n_samples=200, noise=noise_level, test=True)

# Normalization
x_mean, x_std = np.mean(x_train), np.std(x_train)
y_mean, y_std = np.mean(y_train), np.std(y_train)

x_train_norm = (x_train - x_mean) / x_std
y_train_norm = (y_train - y_mean) / y_std
x_val_norm = (x_val - x_mean) / x_std
y_val_norm = (y_val - y_mean) / y_std
x_test_norm = (x_test - x_mean) / x_std
y_test_norm = (y_test - y_mean) / y_std

standardized_noise = noise_level / y_std

# %% [markdown]
# # 2. TAGI Core & Spiking Activation

def initialize_params(layer_dims, seed=None):
    if seed is not None: np.random.seed(seed)
    parameters = {}
    L = len(layer_dims) - 1
    for l in range(L):
        # He-like initialization works better for step functions
        std_init = np.sqrt(2.0 / layer_dims[l])
        mu_w = np.random.normal(0, std_init, (layer_dims[l+1], layer_dims[l]))
        var_w = np.ones((layer_dims[l+1], layer_dims[l])) * (1.0 / layer_dims[l])
        
        mu_b = np.random.normal(0, std_init, (layer_dims[l+1], 1))
        var_b = np.ones((layer_dims[l+1], 1)) * (1.0 / layer_dims[l])
        
        parameters[f'theta_{l}'] = (np.concatenate((mu_w, mu_b), axis=1), 
                                    np.concatenate((var_w, var_b), axis=1))
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

def spiking_activation_forward(Z_l, threshold=0.0):
    """
    The exact probabilistic formulation of a spiking neuron (Heaviside step function).
    """
    mu_z, var_z, _ = Z_l
    epsilon = 1e-6 # Slightly larger epsilon for Bernoulli stability
    
    std_z = np.sqrt(np.maximum(var_z, epsilon))
    alpha = (mu_z - threshold) / std_z
    
    # 1. Expected Firing Rate (Probability of spike)
    mu_a = norm.cdf(alpha)
    
    # 2. Variance of the Spike (Bernoulli: p * (1-p))
    # We clip it at epsilon to prevent division by zero in the backward pass
    var_a = np.maximum(mu_a * (1.0 - mu_a), epsilon)
    
    # 3. Jacobian (Derivative of expected activation w.r.t expected pre-activation)
    T = norm.pdf(alpha) / std_z
    
    return (mu_a, var_a, T)

def model_forward(X_batch, parameters, layer_dims):
    caches = []
    L = len(layer_dims) - 1
    A_prev = (X_batch, np.zeros_like(X_batch))

    for l in range(L - 1):
        theta_l = parameters[f'theta_{l}']
        Z_l = linear_forward(A_prev, theta_l, l)
        
        # USE THE SPIKING ACTIVATION HERE
        A_l = spiking_activation_forward(Z_l, threshold=0.0) 
        
        caches.append({'Z': Z_l, 'A_prev': A_prev, 'theta': theta_l, 'A': A_l})
        A_prev = A_l

    # Final Output Layer (Continuous linear combination of spikes)
    theta_out = parameters[f'theta_{L-1}']
    Z_out = linear_forward(A_prev, theta_out, L-1)
    
    # Output is linear for regression
    T_out = np.ones_like(Z_out[0])
    A_out = (Z_out[0], Z_out[1], T_out)
    caches.append({'Z': Z_out, 'A_prev': A_prev, 'theta': theta_out, 'A': A_out})

    return Z_out, caches

# %% [markdown]
# # 3. Observation & Backward Pass (Identical to your implementation)

def obs_model(Z_out, var_v):
    return (Z_out[0], Z_out[1] + var_v)

def log_likelihood(Y, y_batch):
    mu_y, var_y = Y
    var_y_stable = var_y + 1e-8
    log_ll = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(var_y_stable) - 0.5 * (y_batch - mu_y)**2 / var_y_stable
    return np.sum(log_ll)

def update_output(Y, Z_out, y_batch):
    mu_y, var_y = Y
    mu_z, var_z, _ = Z_out
    K = var_z / (var_y + 1e-8)
    mu_z_y = mu_z + K * (y_batch - mu_y)
    var_z_y = np.maximum(1e-8, var_z * (1 - K))
    return (mu_z_y, var_z_y)

def update_parameters(Z_l, Z_l_y, theta_l):
    mu_z, var_z, cov_z_theta = Z_l
    mu_z_y, var_z_y = Z_l_y
    mu_theta, var_theta = theta_l
    
    J_theta = cov_z_theta / (var_z + 1e-8)[:, np.newaxis, :] 
    delta_mu_z = mu_z_y - mu_z
    delta_var_z = var_z_y - var_z

    delta_mu_theta = np.einsum('kib,kb->kib', J_theta, delta_mu_z)
    delta_var_theta = np.einsum('kib,kb->kib', J_theta**2, delta_var_z)

    avg_delta_mu_theta = np.mean(delta_mu_theta, axis=2)
    avg_delta_var_theta = np.mean(delta_var_theta, axis=2)

    return (mu_theta + avg_delta_mu_theta, np.maximum(1e-8, var_theta + avg_delta_var_theta))

def update_hidden_state(Z_l, Z_l_plus_1, Z_l_plus_1_y, A_l, theta_l_plus_1):
    mu_z, var_z, _ = Z_l
    mu_z_next, var_z_next, _ = Z_l_plus_1
    mu_z_next_y, var_z_next_y = Z_l_plus_1_y
    _, _, T = A_l
    mu_w_next = theta_l_plus_1[0][:, :-1]

    cov_z_next_z = np.einsum('kj,jb,jb->kjb', mu_w_next, var_z, T)
    J_z = cov_z_next_z / (var_z_next + 1e-8)[:, np.newaxis, :]

    delta_mu_z = np.einsum('kji,ki->ji', J_z, mu_z_next_y - mu_z_next)
    delta_var_z = np.einsum('kji,ki->ji', J_z**2, var_z_next_y - var_z_next)

    return (mu_z + delta_mu_z, np.maximum(1e-8, var_z + delta_var_z))

def model_backward(Y, Z_out, y_batch, caches, parameters, layer_dims):
    updated_params = {}
    L = len(layer_dims) - 1
    Z_next_y = update_output(Y, Z_out, y_batch)
    
    for l in reversed(range(L)):
        cache_l = caches[l]
        if l == L - 1:
            updated_params[f'theta_{l}'] = update_parameters(cache_l['Z'], Z_next_y, cache_l['theta'])
        else:
            cache_l_plus_1 = caches[l+1]
            Z_l_y = update_hidden_state(cache_l['Z'], cache_l_plus_1['Z'], Z_next_y, cache_l['A'], parameters[f'theta_{l+1}'])
            updated_params[f'theta_{l}'] = update_parameters(cache_l['Z'], Z_l_y, cache_l['theta'])
            Z_next_y = Z_l_y
            
    return updated_params

# %% [markdown]
# # 4. Training the TAGI-SNN

# Architecture: Wider hidden layers help step functions approximate smooth curves
batch_size = 16
layer_dims = [1, 100, 100, 1] 
parameters = initialize_params(layer_dims, seed=42)
best_params = parameters.copy()

n_epochs = 40 
log_ref = -np.inf

x_train_batch = x_train_norm.reshape(1, -1)
y_train_batch = y_train_norm.reshape(1, -1)
x_val_batch = x_val_norm.reshape(1, -1)
y_val_batch = y_val_norm.reshape(1, -1)
n_samples_train = x_train_batch.shape[1]

print("Training TAGI with Probabilistic Spiking Activations...")
for epoch in tqdm(range(n_epochs)):
    indices = np.random.permutation(n_samples_train)
    x_train_shuffled = x_train_batch[:, indices]
    y_train_shuffled = y_train_batch[:, indices]
    
    for i in range(0, n_samples_train, batch_size):
        end_idx = min(i + batch_size, n_samples_train)
        x_batch_i = x_train_shuffled[:, i:end_idx]
        y_batch_i = y_train_shuffled[:, i:end_idx]
        if x_batch_i.ndim == 1: x_batch_i = x_batch_i.reshape(1, -1)

        Z_out, caches = model_forward(x_batch_i, parameters, layer_dims)
        Y = obs_model(Z_out, standardized_noise**2)
        updated_params = model_backward(Y, Z_out, y_batch_i, caches, parameters, layer_dims)
        parameters.update(updated_params)

    # Validation step
    Z_out_val, _ = model_forward(x_val_batch, parameters, layer_dims)
    Y_val = obs_model(Z_out_val, standardized_noise**2)
    avg_log_ll_val = log_likelihood(Y_val, y_val_batch) / x_val_batch.shape[1]
    
    if avg_log_ll_val > log_ref:
        log_ref = avg_log_ll_val
        best_params = {k: (v[0].copy(), v[1].copy()) for k, v in parameters.items()}

# %% [markdown]
# # 5. Evaluation and Plotting

# Test pass
x_test_batch = x_test_norm.reshape(1, -1)
Z_out_test, _ = model_forward(x_test_batch, best_params, layer_dims)
Y_test = obs_model(Z_out_test, standardized_noise**2)

mu_y_test, var_y_test = Y_test
y_pred_orig = (mu_y_test.flatten() * y_std) + y_mean
var_pred_orig = var_y_test.flatten() * (y_std**2)

# Sort for smooth plotting
sort_idx = np.argsort(x_test)
x_plot = x_test[sort_idx]
y_pred_plot = y_pred_orig[sort_idx]
std_pred_plot = np.sqrt(var_pred_orig[sort_idx])

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='gray', alpha=0.3, label='Noisy Train Data', s=15)
plt.plot(x_true, y_true, color='black', linewidth=2, linestyle='--', label='True Sine Wave')
plt.plot(x_plot, y_pred_plot, color='blue', linewidth=2, label='TAGI-Spike Prediction')
plt.fill_between(x_plot, y_pred_plot - 2*std_pred_plot, y_pred_plot + 2*std_pred_plot, 
                 color='blue', alpha=0.2, label=r'$\pm 2\sigma$ Uncertainty')

plt.title('TAGI Probabilistic Spiking Network - Sine Wave Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print(f"Final MSE on Test Set: {mean_squared_error(y_test, y_pred_orig):.4f}")