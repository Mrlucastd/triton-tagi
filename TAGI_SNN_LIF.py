# %%
"""
TAGI_SNN_LIF.py

Extension of TAGI_SNN.py. Three dynamics parameters are now fully inferred
by TAGI instead of being fixed hyperparameters:

  1. ADAPTIVE T  (Riccati stopping)
     Stop when  max |σ²_V(t) − σ²_V(t-1)| < ε.
     Once the membrane variance stabilises, the Kalman gain is constant and
     further timesteps add zero information to the TAGI update.

  2. LEARNABLE β PER NEURON
     V(t) = β · carry(t-1) + z(t)   is linear in β  →  TAGI inference.
       μ_V(t)  = μ_β · carry  +  μ_z
       σ²_V(t) = σ²_β · carry²  +  μ_β² · σ²_carry  +  σ²_z
       cov(V, β) = σ²_β · carry            K_β = cov(V,β)/σ²_V

  3. LEARNABLE θ PER NEURON  (spike threshold)
     θ enters as a negative shift before the spike:
       z_eff = V − θ,   α = z_eff / σ_eff,   σ_eff = √(σ²_V + σ²_θ)

     θ is like a negative bias with cov(z_eff, θ) = −σ²_θ:
       K_θ = −σ²_θ / σ²_eff             (negative gain — right sign)

     If the network needs more spikes (V_y > V), K_θ < 0 → threshold drops
     → neurons become more excitable.  Exact homeostatic behaviour.

Only fixed hyperparameters remaining: max_T (safety ceiling) and ε_conv.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

np.random.seed(42)

# %% [markdown]
# # 1. Dataset (same as TAGI_SNN.py)

def get_data(n_samples=1000, noise=3, name='linear', test=False):
    if test:
        if name == 'linear':
            x = np.linspace(-5, 5, n_samples)
            y = x**3 + np.random.randn(len(x)) * noise
            return (x, y), (x, x**3)
    if name == 'linear':
        x = np.linspace(-5, 5, n_samples)
        y = x**3 + np.random.randn(len(x)) * noise
        return x, y

noise_level = 9
x_train, y_train = get_data(n_samples=800, noise=noise_level)
x_val,   y_val   = get_data(n_samples=200, noise=noise_level)
(x_test, y_test), (x_true, y_true) = get_data(n_samples=200, noise=noise_level, test=True)

x_mean, x_std = np.mean(x_train), np.std(x_train)
y_mean, y_std = np.mean(y_train), np.std(y_train)

x_train_norm = (x_train - x_mean) / x_std
y_train_norm = (y_train - y_mean) / y_std
x_val_norm   = (x_val   - x_mean) / x_std
y_val_norm   = (y_val   - y_mean) / y_std
x_test_norm  = (x_test  - x_mean) / x_std

standardized_noise = noise_level / y_std

# %% [markdown]
# # 2. Parameter Initialisation (weights + β + θ per hidden layer)

def initialize_params(layer_dims,
                      beta_init=0.9,   beta_var_init=0.05,
                      thresh_init=0.5, thresh_var_init=0.05,
                      seed=None):
    """
    Initialises:
      theta_l  : (μ_W|b, σ²_W|b)  weights & biases for all layers
      beta_l   : (μ_β,   σ²_β)    per-neuron leak factor, hidden layers only
      thresh_l : (μ_θ,   σ²_θ)    per-neuron spike threshold, hidden layers only

    Prior variances for β and θ are kept tight (0.05) so parameter uncertainty
    does not dominate σ²_eff before learning starts.
    """
    if seed is not None: np.random.seed(seed)
    parameters = {}
    L = len(layer_dims) - 1

    for l in range(L):
        std_init = np.sqrt(2.0 / layer_dims[l])
        mu_w  = np.random.normal(0, std_init, (layer_dims[l+1], layer_dims[l]))
        var_w = np.ones((layer_dims[l+1], layer_dims[l])) / layer_dims[l]
        mu_b  = np.random.normal(0, std_init, (layer_dims[l+1], 1))
        var_b = np.ones((layer_dims[l+1], 1)) / layer_dims[l]
        parameters[f'theta_{l}'] = (
            np.concatenate((mu_w, mu_b), axis=1),
            np.concatenate((var_w, var_b), axis=1)
        )

    for l in range(L - 1):
        n = layer_dims[l + 1]
        parameters[f'beta_{l}']   = (np.full(n, beta_init),   np.full(n, beta_var_init))
        parameters[f'thresh_{l}'] = (np.full(n, thresh_init), np.full(n, thresh_var_init))

    return parameters


# %% [markdown]
# # 3. TAGI Core (unchanged from TAGI_SNN.py)

def linear_forward(A_prev, theta_l):
    mu_a, var_a = A_prev[0], A_prev[1]
    mu_theta, var_theta = theta_l
    mu_w,  mu_b  = mu_theta[:, :-1], mu_theta[:, -1:]
    var_w, var_b = var_theta[:, :-1], var_theta[:, -1:]

    mu_z  = mu_w @ mu_a + mu_b
    var_z = (var_w @ var_a
             + var_w @ (mu_a**2)
             + (mu_w**2) @ var_a
             + var_b)

    B           = mu_a.shape[1]
    cov_z_w     = np.einsum('ji,kj->kji', mu_a, var_w)
    cov_z_b     = np.repeat(var_b[:, :, np.newaxis], B, axis=2)
    cov_z_theta = np.concatenate((cov_z_w, cov_z_b), axis=1)

    return mu_z, var_z, cov_z_theta


def obs_model(Z_out, var_v):
    return Z_out[0], Z_out[1] + var_v


def log_likelihood(Y, y_batch):
    mu_y, var_y = Y
    v = var_y + 1e-8
    return np.sum(-0.5*np.log(2*np.pi) - 0.5*np.log(v) - 0.5*(y_batch - mu_y)**2 / v)


def update_output(Y, Z_out, y_batch):
    mu_y, var_y = Y
    mu_z, var_z, _ = Z_out
    K       = var_z / (var_y + 1e-8)
    mu_z_y  = mu_z + K * (y_batch - mu_y)
    var_z_y = np.maximum(1e-8, var_z * (1 - K))
    return mu_z_y, var_z_y


def update_parameters(Z_l, Z_l_y, theta_l):
    mu_z, var_z, cov_z_theta = Z_l
    mu_z_y, var_z_y = Z_l_y
    mu_theta, var_theta = theta_l

    J          = cov_z_theta / (var_z + 1e-8)[:, np.newaxis, :]
    dmu_theta  = np.mean(np.einsum('kib,kb->kib', J,    mu_z_y  - mu_z),  axis=2)
    dvar_theta = np.mean(np.einsum('kib,kb->kib', J**2, var_z_y - var_z), axis=2)

    return mu_theta + dmu_theta, np.maximum(1e-8, var_theta + dvar_theta)


def update_hidden_state(Z_l, Z_l_plus_1, Z_l_plus_1_y, A_l, theta_l_plus_1):
    mu_z,      var_z,      _  = Z_l
    mu_z_next, var_z_next, _  = Z_l_plus_1
    mu_z_next_y, var_z_next_y = Z_l_plus_1_y
    _, _, T   = A_l
    mu_w_next = theta_l_plus_1[0][:, :-1]

    cov = np.einsum('kj,jb,jb->kjb', mu_w_next, var_z, T)
    J   = cov / (var_z_next + 1e-8)[:, np.newaxis, :]

    dmu  = np.einsum('kji,ki->ji', J,    mu_z_next_y - mu_z_next)
    dvar = np.einsum('kji,ki->ji', J**2, var_z_next_y - var_z_next)

    return mu_z + dmu, np.maximum(1e-8, var_z + dvar)


# %% [markdown]
# # 4. Probabilistic LIF with Learnable β and θ

def lif_forward(V_state, Z_synapse, beta_param, thresh_param):
    """
    One-step probabilistic LIF with β and θ both as Gaussian variables.

    Spike threshold enters via the effective pre-activation:
        z_eff  = V − θ
        σ²_eff = σ²_V + σ²_θ        (threshold uncertainty widens the distribution)
        α      = (μ_V − μ_θ) / √σ²_eff

    This is used for BOTH the previous-step soft-reset and the current spike.

    cov(z_eff, θ) = −σ²_θ   →   K_θ = −σ²_θ / σ²_eff   (stored for backward)

    Args
    ----
    V_state     : (μ_V_prev, σ²_V_prev)
    Z_synapse   : (μ_z, σ²_z, cov_z_θ_weights)   from linear_forward
    beta_param  : (μ_β,  σ²_β)   per-neuron, shape (n,)
    thresh_param: (μ_θ,  σ²_θ)   per-neuron, shape (n,)

    Returns
    -------
    V_state_new : (μ_V_new, σ²_V_new)
    Z_lif       : (μ_V_new, σ²_V_new, cov_z_θ_weights)   "pre-activation"
    A_lif       : (μ_s, σ²_s, T_V)                        "post-activation"
    carry       : μ_V(t-1)·(1−μ_s(t-1))                  for β backward
    cov_V_beta  : σ²_β · carry                             for β backward
    cov_V_thresh: −σ²_θ  (scalar per neuron, batch mean)   for θ backward
    """
    mu_V_prev, var_V_prev = V_state
    mu_z, var_z, cov_z_weights = Z_synapse
    mu_beta,   var_beta   = beta_param[0].reshape(-1, 1),   beta_param[1].reshape(-1, 1)
    mu_thresh, var_thresh = thresh_param[0].reshape(-1, 1), thresh_param[1].reshape(-1, 1)
    eps = 1e-6

    # --- Previous spike with learnable threshold (drives soft reset) ---
    sigma_eff_prev_sq = np.maximum(var_V_prev + var_thresh, eps)
    alpha_prev = (mu_V_prev - mu_thresh) / np.sqrt(sigma_eff_prev_sq)
    mu_s_prev  = norm.cdf(alpha_prev)
    var_s_prev = np.maximum(mu_s_prev * (1.0 - mu_s_prev), eps)

    # --- Carry term and new membrane potential ---
    carry     = mu_V_prev * (1.0 - mu_s_prev)
    var_carry = var_V_prev * (1.0 - mu_s_prev)**2 + mu_V_prev**2 * var_s_prev

    mu_V_new  = mu_beta * carry + mu_z
    var_V_new = np.maximum(var_beta * carry**2 + mu_beta**2 * var_carry + var_z, eps)

    # --- Current spike with learnable threshold ---
    sigma_eff_sq = np.maximum(var_V_new + var_thresh, eps)
    alpha_new    = (mu_V_new - mu_thresh) / np.sqrt(sigma_eff_sq)
    mu_s_new     = norm.cdf(alpha_new)
    var_s_new    = np.maximum(mu_s_new * (1.0 - mu_s_new), eps)
    T_V          = norm.pdf(alpha_new) / np.sqrt(sigma_eff_sq)   # dμ_s / dμ_V

    # --- Covariances for parameter backward passes ---
    cov_V_beta   = var_beta * carry                  # cov(V, β)   shape (n, B)
    cov_V_thresh = -var_thresh / sigma_eff_sq        # K_θ directly: −σ²_θ / σ²_eff (n, B)

    V_state_new = (mu_V_new, var_V_new)
    Z_lif = (mu_V_new, var_V_new, cov_z_weights)
    A_lif = (mu_s_new, var_s_new, T_V)

    return V_state_new, Z_lif, A_lif, carry, cov_V_beta, cov_V_thresh


def update_beta(V_state, V_state_y, cov_V_beta, beta_param):
    """
    TAGI update for per-neuron β.
        K_β = σ²_β · carry / σ²_V   (cov_V_beta / σ²_V)
    β clamped to [0.01, 0.99].
    """
    mu_V, var_V     = V_state
    mu_V_y, var_V_y = V_state_y
    mu_beta, var_beta = beta_param

    K_beta         = cov_V_beta / (var_V + 1e-8)
    delta_mu_beta  = np.mean(K_beta    * (mu_V_y  - mu_V),  axis=1)
    delta_var_beta = np.mean(K_beta**2 * (var_V_y - var_V), axis=1)

    return (np.clip(mu_beta + delta_mu_beta, 0.01, 0.99),
            np.maximum(1e-8, var_beta + delta_var_beta))


def update_thresh(V_state, V_state_y, cov_V_thresh, thresh_param):
    """
    TAGI update for per-neuron threshold θ.

    cov(z_eff, θ) = −σ²_θ,  so  K_θ = −σ²_θ / σ²_eff  (already stored in cov_V_thresh).

    Δμ_θ = mean_batch( K_θ · (μ_V_y − μ_V) )    ← negative gain: more spikes needed
                                                     → threshold decreases. Correct.
    θ is unconstrained (can be negative for very active neurons).
    """
    mu_V, var_V     = V_state
    mu_V_y, var_V_y = V_state_y
    mu_thresh, var_thresh = thresh_param

    # cov_V_thresh already equals K_θ = −σ²_θ / σ²_eff
    K_thresh         = cov_V_thresh
    delta_mu_thresh  = np.mean(K_thresh    * (mu_V_y  - mu_V),  axis=1)
    delta_var_thresh = np.mean(K_thresh**2 * (var_V_y - var_V), axis=1)

    return (mu_thresh + delta_mu_thresh,
            np.maximum(1e-8, var_thresh + delta_var_thresh))


# %% [markdown]
# # 5. Adaptive Forward Pass (Riccati stopping)

def _init_voltage_ss(layer_dims, parameters, X_input, n_iter=100):
    """
    Find the TRUE (μ_V*, σ²_V*) fixed point per neuron via iteration.

    After training, neurons specialise into two regimes:
      - "active" neurons  p* → 1  →  carry ≈ 0    →  V* ≈ z      (fast)
      - "silent" neurons  p* → 0  →  carry ≈ V*   →  V* = z/(1-β) (slow!)

    For silent neurons with μ_β=0.9 the warm-start μ_V=μ_θ can be 20+ units
    from the true fixed point, and convergence rate is 0.9/step.
    After 30 steps the error is still ~0.86 — nowhere near converged.

    Fix: use an analytical two-regime warm-start:
        silent: V*_warm = μ_z / (1 − μ_β)   (carry = V*, p* ≈ 0)
        active: V*_warm = μ_z                (carry = 0,  p* ≈ 1)
    Pick whichever produces an α closer to zero (i.e. more self-consistent).
    Then run 100 iterations to polish the estimate.

    0.9^100 ≈ 2.7e-5, so even the worst-case silent neuron converges to
    ≪ 0.1% error, ensuring the forward Riccati loop starts at its fixed
    point and converges in 1 step.
    """
    L   = len(layer_dims) - 1
    eps = 1e-6

    V_states      = []
    var_V_ss_list = []

    var_a = np.zeros_like(X_input)   # layer 0: deterministic input
    mu_a  = X_input

    for l in range(L - 1):
        mu_w      = parameters[f'theta_{l}'][0][:, :-1]
        var_w     = parameters[f'theta_{l}'][1][:, :-1]
        mu_b_w    = parameters[f'theta_{l}'][0][:, -1:]   # bias mean
        var_b     = parameters[f'theta_{l}'][1][:, -1:]
        mu_beta   = parameters[f'beta_{l}'][0].reshape(-1, 1)
        var_beta  = parameters[f'beta_{l}'][1].reshape(-1, 1)
        mu_theta  = parameters[f'thresh_{l}'][0].reshape(-1, 1)
        var_theta = parameters[f'thresh_{l}'][1].reshape(-1, 1)

        # Synaptic statistics — fixed for this layer / batch
        mu_z       = mu_w @ mu_a + mu_b_w                            # (n, B)
        sigma_z_sq = (var_w @ var_a
                      + var_w @ (mu_a ** 2)
                      + (mu_w ** 2) @ var_a
                      + var_b)                                        # (n, B)

        # Smart two-regime warm-start
        # silent regime: V* = μ_z/(1-β)   (when p*≈0, carry=V*)
        # active regime: V* = μ_z         (when p*≈1, carry≈0)
        mu_V_silent = mu_z / np.maximum(1.0 - mu_beta, 1e-3)
        mu_V_active = mu_z
        # Self-consistency: pick the candidate whose α = (V*-θ)/σ_eff matches
        # the assumed regime (silent→α≪0, active→α≫0).
        # Equivalently, pick the one closer to the threshold (more balanced).
        sigma_eff_init = np.sqrt(np.maximum(sigma_z_sq + var_theta, eps))
        alpha_silent = (mu_V_silent - mu_theta) / sigma_eff_init
        alpha_active = (mu_V_active - mu_theta) / sigma_eff_init
        use_silent   = np.abs(alpha_silent) < np.abs(alpha_active)
        mu_V = np.where(use_silent, mu_V_silent, mu_V_active)

        # Variance warm-start (p*=0.5 analytical formula as fallback)
        denom      = np.maximum(1.0 - mu_beta ** 2 * 0.25, eps)
        sigma_V_sq = np.maximum(
            ((var_beta + mu_beta ** 2) * mu_theta ** 2 * 0.25 + sigma_z_sq) / denom,
            1e-4
        )

        # Fixed-point iteration for the TRUE (μ_V*, σ²_V*)
        # 100 iterations → error < 0.003% even for the worst silent neuron (β=0.99)
        for _ in range(n_iter):
            sigma_eff_sq = np.maximum(sigma_V_sq + var_theta, eps)
            alpha        = (mu_V - mu_theta) / np.sqrt(sigma_eff_sq)
            p            = norm.cdf(alpha)
            var_s        = np.maximum(p * (1.0 - p), eps)

            carry        = mu_V * (1.0 - p)
            var_carry    = sigma_V_sq * (1.0 - p) ** 2 + mu_V ** 2 * var_s

            mu_V       = mu_beta * carry + mu_z
            sigma_V_sq = np.maximum(
                var_beta * carry ** 2 + mu_beta ** 2 * var_carry + sigma_z_sq,
                1e-4
            )

        V_states.append((mu_V, sigma_V_sq))
        var_V_ss_list.append(sigma_V_sq.copy())

        # Propagate ACTUAL activation statistics to the next layer
        sigma_eff_sq = np.maximum(sigma_V_sq + var_theta, eps)
        alpha        = (mu_V - mu_theta) / np.sqrt(sigma_eff_sq)
        p_ss         = norm.cdf(alpha)
        var_a        = np.maximum(p_ss * (1.0 - p_ss), eps)
        mu_a         = p_ss

    return V_states, var_V_ss_list


def model_forward_adaptive(X_input, parameters, layer_dims,
                            max_T=50, min_T=5, eps_conv=1e-4):
    """
    Forward pass that stops when membrane variance converges.

    Always runs at least min_T steps to ensure meaningful temporal integration
    (the fixed-point init places us at steady state, so without min_T the
    Riccati criterion would trigger after just 1 step).
    Stops early once max |Δσ²_V|/σ²_V < eps_conv.
    All of β, θ are read from `parameters` — no fixed hyperparameters.
    """
    L = len(layer_dims) - 1
    B = X_input.shape[1]

    # Start at the analytical Riccati fixed point — eliminates the slow transient
    V_states, var_V_ss_list = _init_voltage_ss(layer_dims, parameters, X_input)
    # Seed the convergence tracker at the same value so the first delta ≈ 0
    var_V_prev_layers = [v.copy() for v in var_V_ss_list]

    mu_out_sum  = np.zeros((layer_dims[-1], B))
    var_out_sum = np.zeros((layer_dims[-1], B))
    cov_out_sum = None
    all_caches  = []
    T_actual    = 0

    for t in range(max_T):
        A_prev   = (X_input, np.zeros_like(X_input))
        caches_t = []
        new_V_states = []

        for l in range(L - 1):
            theta_l      = parameters[f'theta_{l}']
            beta_param   = parameters[f'beta_{l}']
            thresh_param = parameters[f'thresh_{l}']
            Z_syn        = linear_forward(A_prev, theta_l)
            V_new, Z_lif, A_lif, carry, cov_V_beta, cov_V_thresh = lif_forward(
                V_states[l], Z_syn, beta_param, thresh_param
            )
            caches_t.append({
                'Z':            Z_lif,
                'A':            A_lif,
                'A_prev':       A_prev,
                'theta':        theta_l,
                'carry':        carry,
                'cov_V_beta':   cov_V_beta,
                'cov_V_thresh': cov_V_thresh,
            })
            new_V_states.append(V_new)
            A_prev = A_lif

        V_states = new_V_states
        T_actual += 1

        theta_out = parameters[f'theta_{L-1}']
        Z_out_t   = linear_forward(A_prev, theta_out)
        T_out     = np.ones_like(Z_out_t[0])
        caches_t.append({
            'Z': Z_out_t, 'A': (Z_out_t[0], Z_out_t[1], T_out),
            'A_prev': A_prev, 'theta': theta_out,
        })
        all_caches.append(caches_t)

        mu_out_sum  += Z_out_t[0]
        var_out_sum += Z_out_t[1]
        cov_out_sum  = Z_out_t[2] if cov_out_sum is None else cov_out_sum + Z_out_t[2]

        # Riccati stopping: RELATIVE variance change across all hidden layers.
        # Absolute tolerance was wrong: σ²_V ≈ 2.5, so |Δσ²_V|/σ²_V < eps_conv
        # is the right criterion (eps_conv ≈ 0.01 means <1% relative change).
        max_delta = 0.0
        for l in range(L - 1):
            var_V_new_l  = V_states[l][1]
            mean_new     = var_V_new_l.mean(axis=1)
            mean_prev    = var_V_prev_layers[l].mean(axis=1)
            rel_delta_l  = np.max(np.abs(mean_new - mean_prev) / (mean_prev + 1e-8))
            max_delta    = max(max_delta, rel_delta_l)
            var_V_prev_layers[l] = var_V_new_l.copy()

        if max_delta < eps_conv and T_actual >= min_T:
            break

    Z_out_avg = (mu_out_sum / T_actual, var_out_sum / T_actual, cov_out_sum / T_actual)
    return Z_out_avg, all_caches, T_actual


# %% [markdown]
# # 6. Adaptive Backward Pass (updates weights, β, and θ)

def model_backward_adaptive(Y, Z_out_avg, y_batch, all_caches,
                             parameters, layer_dims):
    """
    TAGI backward pass that infers θ (weights/biases), β (leak), and θ_spike (threshold).
    Independent-timestep approximation: deltas are averaged over T_actual steps.
    """
    L        = len(layer_dims) - 1
    T_actual = len(all_caches)

    Z_next_y_avg = update_output(Y, Z_out_avg, y_batch)

    delta_mu_w  = {f'theta_{l}':  np.zeros_like(parameters[f'theta_{l}'][0])  for l in range(L)}
    delta_var_w = {f'theta_{l}':  np.zeros_like(parameters[f'theta_{l}'][1])  for l in range(L)}
    delta_mu_b  = {f'beta_{l}':   np.zeros(layer_dims[l+1])                   for l in range(L-1)}
    delta_var_b = {f'beta_{l}':   np.zeros(layer_dims[l+1])                   for l in range(L-1)}
    delta_mu_t  = {f'thresh_{l}': np.zeros(layer_dims[l+1])                   for l in range(L-1)}
    delta_var_t = {f'thresh_{l}': np.zeros(layer_dims[l+1])                   for l in range(L-1)}

    for t in range(T_actual):
        caches_t   = all_caches[t]
        Z_next_y_t = Z_next_y_avg

        for l in reversed(range(L)):
            cache_l = caches_t[l]
            theta_l = parameters[f'theta_{l}']
            mu_t, var_t = theta_l

            if l == L - 1:
                new_mu, new_var = update_parameters(cache_l['Z'], Z_next_y_t, theta_l)
            else:
                cache_next = caches_t[l + 1]
                Z_l_y = update_hidden_state(
                    cache_l['Z'], cache_next['Z'], Z_next_y_t,
                    cache_l['A'], parameters[f'theta_{l+1}']
                )
                new_mu, new_var = update_parameters(cache_l['Z'], Z_l_y, theta_l)

                V_state = (cache_l['Z'][0], cache_l['Z'][1])

                # β update
                new_mu_b, new_var_b = update_beta(
                    V_state, Z_l_y, cache_l['cov_V_beta'], parameters[f'beta_{l}']
                )
                delta_mu_b[f'beta_{l}']  += new_mu_b  - parameters[f'beta_{l}'][0]
                delta_var_b[f'beta_{l}'] += new_var_b - parameters[f'beta_{l}'][1]

                # θ update
                new_mu_th, new_var_th = update_thresh(
                    V_state, Z_l_y, cache_l['cov_V_thresh'], parameters[f'thresh_{l}']
                )
                delta_mu_t[f'thresh_{l}']  += new_mu_th  - parameters[f'thresh_{l}'][0]
                delta_var_t[f'thresh_{l}'] += new_var_th - parameters[f'thresh_{l}'][1]

                Z_next_y_t = Z_l_y

            delta_mu_w[f'theta_{l}']  += new_mu  - mu_t
            delta_var_w[f'theta_{l}'] += new_var - var_t

    updated = {}
    for l in range(L):
        mu_old, var_old = parameters[f'theta_{l}']
        updated[f'theta_{l}'] = (
            mu_old + delta_mu_w[f'theta_{l}']  / T_actual,
            np.maximum(1e-8, var_old + delta_var_w[f'theta_{l}'] / T_actual)
        )
    for l in range(L - 1):
        mu_old_b,  var_old_b  = parameters[f'beta_{l}']
        mu_old_th, var_old_th = parameters[f'thresh_{l}']
        updated[f'beta_{l}'] = (
            np.clip(mu_old_b + delta_mu_b[f'beta_{l}'] / T_actual, 0.01, 0.99),
            np.maximum(1e-8, var_old_b + delta_var_b[f'beta_{l}'] / T_actual)
        )
        updated[f'thresh_{l}'] = (
            mu_old_th + delta_mu_t[f'thresh_{l}'] / T_actual,
            np.maximum(1e-8, var_old_th + delta_var_t[f'thresh_{l}'] / T_actual)
        )

    return updated


# %% [markdown]
# # 7. Training

max_T    = 50    # safety ceiling
min_T    = 5     # minimum timesteps — ensures temporal integration happens
eps_conv = 0.001 # Riccati convergence: relative |Δσ²_V| / σ²_V < 0.1%

batch_size = 16
layer_dims = [1, 100, 100, 1]
parameters = initialize_params(layer_dims,
                               beta_init=0.9,   beta_var_init=0.1,
                               thresh_init=0.5, thresh_var_init=0.1,
                               seed=42)
best_params = parameters.copy()

n_epochs = 40
log_ref  = -np.inf

x_train_b = x_train_norm.reshape(1, -1)
y_train_b = y_train_norm.reshape(1, -1)
x_val_b   = x_val_norm.reshape(1, -1)
y_val_b   = y_val_norm.reshape(1, -1)
N_train   = x_train_b.shape[1]

L_hidden = len(layer_dims) - 2   # number of hidden layers

t_history          = []
beta_var_history   = []   # mean σ²_β per epoch  — should shrink as β is learned
thresh_var_history = []   # mean σ²_θ per epoch  — should shrink as θ is learned

print(f"Training TAGI-LIF  (max_T={max_T}, ε_rel={eps_conv})")
print("β, θ, and T are all inferred — no fixed dynamics hyperparameters.")

for epoch in tqdm(range(n_epochs)):
    idx = np.random.permutation(N_train)
    xs, ys = x_train_b[:, idx], y_train_b[:, idx]
    epoch_T = []

    for i in range(0, N_train, batch_size):
        j = min(i + batch_size, N_train)
        Z_avg, caches, T_actual = model_forward_adaptive(
            xs[:, i:j], parameters, layer_dims, max_T=max_T, min_T=min_T, eps_conv=eps_conv
        )
        Y = obs_model(Z_avg, standardized_noise**2)
        parameters.update(
            model_backward_adaptive(Y, Z_avg, ys[:, i:j], caches, parameters, layer_dims)
        )
        epoch_T.append(T_actual)

    t_history.append(np.mean(epoch_T))

    # Track mean posterior variance of β and θ across all hidden layers
    beta_var_history.append(
        np.mean([parameters[f'beta_{l}'][1].mean() for l in range(L_hidden)])
    )
    thresh_var_history.append(
        np.mean([parameters[f'thresh_{l}'][1].mean() for l in range(L_hidden)])
    )

    Z_val, _, _ = model_forward_adaptive(
        x_val_b, parameters, layer_dims, max_T=max_T, min_T=min_T, eps_conv=eps_conv
    )
    Y_val  = obs_model(Z_val, standardized_noise**2)
    ll_val = log_likelihood(Y_val, y_val_b) / x_val_b.shape[1]
    if ll_val > log_ref:
        log_ref     = ll_val
        best_params = {k: (v[0].copy(), v[1].copy()) for k, v in parameters.items()}

# Report learned parameters
print("\n--- Learned dynamics (per hidden layer) ---")
for l in range(L_hidden):
    mu_b,  var_b  = best_params[f'beta_{l}']
    mu_th, var_th = best_params[f'thresh_{l}']
    print(f"  Layer {l}:")
    print(f"    β:  μ={mu_b.mean():.4f} ± {mu_b.std():.4f}   "
          f"σ²_β (mean)={var_b.mean():.5f}  (init=0.1)")
    print(f"    θ:  μ={mu_th.mean():.4f} ± {mu_th.std():.4f}   "
          f"σ²_θ (mean)={var_th.mean():.5f}  (init=0.1)")

print(f"\nAvg T per epoch (first→last): {t_history[0]:.1f} → {t_history[-1]:.1f}")


# %% [markdown]
# # 8. Evaluation

x_test_b          = x_test_norm.reshape(1, -1)
Z_test, _, T_test = model_forward_adaptive(
    x_test_b, best_params, layer_dims, max_T=max_T, min_T=min_T, eps_conv=eps_conv
)
Y_test = obs_model(Z_test, standardized_noise**2)

mu_y_test, var_y_test = Y_test
y_pred_orig   = mu_y_test.flatten() * y_std + y_mean
std_pred_orig = np.sqrt(var_y_test.flatten()) * y_std

sort_idx      = np.argsort(x_test)
x_plot        = x_test[sort_idx]
y_pred_plot   = y_pred_orig[sort_idx]
std_pred_plot = std_pred_orig[sort_idx]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# --- (0,0) Prediction ---
ax = axes[0, 0]
ax.scatter(x_train, y_train, color='gray', alpha=0.3, label='Noisy Train Data', s=15)
ax.plot(x_true, y_true, 'k--', linewidth=2, label='True y = x³')
ax.plot(x_plot, y_pred_plot, color='royalblue', linewidth=2,
        label=f'TAGI-LIF Adaptive  (T={T_test})')
ax.fill_between(x_plot, y_pred_plot - 2*std_pred_plot, y_pred_plot + 2*std_pred_plot,
                color='royalblue', alpha=0.2, label=r'$\pm 2\sigma$')
ax.set_title('TAGI-LIF — β, θ, T all inferred')
ax.set_xlabel('X'); ax.set_ylabel('Y')
ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)

# --- (0,1) T per epoch ---
ax2 = axes[0, 1]
ax2.plot(t_history, color='darkorange', linewidth=2)
ax2.axhline(max_T, color='red', linestyle='--', alpha=0.5, label=f'max_T={max_T}')
ax2.set_title('Avg T used per epoch (Riccati stopping)')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Avg timesteps')
ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6)

# --- (1,0) β posterior variance ---
ax3 = axes[1, 0]
ax3.plot(beta_var_history, color='steelblue', linewidth=2)
ax3.axhline(0.1, color='gray', linestyle='--', alpha=0.6, label='prior σ²_β = 0.1')
ax3.set_title('Mean σ²_β over training\n(shrinking = β is being learned)')
ax3.set_xlabel('Epoch'); ax3.set_ylabel('Mean posterior σ²_β')
ax3.legend(); ax3.grid(True, linestyle='--', alpha=0.6)

# --- (1,1) θ posterior variance ---
ax4 = axes[1, 1]
ax4.plot(thresh_var_history, color='tomato', linewidth=2)
ax4.axhline(0.1, color='gray', linestyle='--', alpha=0.6, label='prior σ²_θ = 0.1')
ax4.set_title('Mean σ²_θ over training\n(shrinking = θ is being learned)')
ax4.set_xlabel('Epoch'); ax4.set_ylabel('Mean posterior σ²_θ')
ax4.legend(); ax4.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

print(f"Final MSE: {mean_squared_error(y_test, y_pred_orig):.4f}")
print(f"Test inference: T = {T_test} timesteps")
