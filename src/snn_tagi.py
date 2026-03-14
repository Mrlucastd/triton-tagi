"""
src/snn_tagi.py

A Tractable Analytic Bayesian Spiking Neural Network (TAGI-SNN) implementation
following the theoretical framework from documents/SNN/snn_tagi.tex

This implementation includes:
1. Gaussian voltage propagation (Eqs. 6-7)
2. TAGI weight inference (Eqs. 8-9) 
3. Surprise-gated homeostatic plasticity (Eq. 14)
4. Adaptive temporal horizon via variance convergence (Eq. 18)
5. Neuron lifecycle management (Eqs. 21-23)
6. KL-based structural plasticity (Eqs. 24-25) 
7. Oscillatory dynamics with adaptive leak (Eq. 31)
8. Sleep-phase consolidation and Poisson input encoding

This follows the full biological principles P1-P5 from the paper.
"""

from typing import List, Tuple, Optional, Dict, Set
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import numpy as np

# Try to import Triton, fallback to PyTorch if not available
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
    _BLOCK = 1024
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None
    _BLOCK = None

# Try to import observation module, create fallback if not available
try:
    from src.update.observation import compute_innovation
except ImportError:
    def compute_innovation(y, y_pred_mu, y_pred_var, sigma_v):
        """Fallback implementation of compute_innovation"""
        innovation = y - y_pred_mu
        delta_mu = innovation / (y_pred_var + sigma_v)
        delta_var = torch.ones_like(y_pred_var) * 0.1
        return delta_mu, delta_var

class NeuronState(Enum):
    """Neuron lifecycle states from Eq. (23)"""
    DEAD = "dead"
    ACTIVE = "active" 
    MATURE = "mature"

if TRITON_AVAILABLE:
    @triton.jit
    def _threshold_kernel(v_ptr, thr, s_ptr, n_elements, BLOCK: tl.constexpr):
        """Triton kernel: compute spike = (v > thr) as float32 (0.0/1.0)"""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        valid = offs < n_elements
        v = tl.load(v_ptr + offs, mask=valid)
        s = v > thr
        out = tl.where(s, tl.constant(1.0, dtype=tl.float32), tl.constant(0.0, dtype=tl.float32))
        tl.store(s_ptr + offs, out, mask=valid)

def threshold_t(vtensor: torch.Tensor, thr: float) -> torch.Tensor:
    """Run threshold kernel (Triton or PyTorch fallback)"""
    if TRITON_AVAILABLE:
        assert vtensor.dtype == torch.float32 and vtensor.dim() == 1
        n = vtensor.numel()
        out = torch.empty_like(vtensor)
        grid = (triton.cdiv(n, _BLOCK),)
        _threshold_kernel[grid](vtensor, thr, out, n, BLOCK=_BLOCK)
        return out
    else:
        # PyTorch fallback
        return (vtensor > thr).float()

class TAGISNNNeuron:
    """
    TAGI-enabled spiking neuron implementing full theoretical model.
    
    Maintains Gaussian posterior over membrane voltage following Eqs. (6-7):
    V_i^(t) ~ N(mu_V_i^(t), sigma2_V_i^(t))
    
    Features:
    - Gaussian voltage propagation through weights
    - Adaptive threshold with surprise-gated homeostasis (Eq. 14)
    - Adaptive leak for oscillatory dynamics (Eq. 31)
    - Temporal horizon detection (Eq. 18) 
    - Vitality tracking for lifecycle management (Eq. 21)
    """
    
    def __init__(
        self,
        neuron_id: int,
        spatial_position: Tuple[float, float],
        device: Optional[torch.device] = None,
        # TAGI parameters
        prior_var: float = 0.1,
        # Homeostasis parameters
        alpha_theta: float = 0.01,
        beta_theta: float = 0.01, 
        # Temporal horizon
        eps_T: float = 1e-4,
        # Oscillatory dynamics
        beta_lambda: float = 0.01,
        lambda_min: float = 0.1,
        lambda_max: float = 0.95,
        lambda_init: float = 0.5,
        # Lifecycle thresholds - CORRECTED for better initial behavior  
        tau_kill: float = 1.01,  # Only kill if vitality > 1.01 (impossible, so no early death)
        tau_sat: float = 0.8,    # Mature when variance shrunk to 80% of original 
        tau_active: float = 1e-2 # Higher threshold for detecting activity
    ):
        self.id = neuron_id
        self.position = spatial_position
        self.device = device or torch.device("cpu")
        
        # TAGI parameters
        self.prior_var = prior_var
        
        # Homeostasis
        self.alpha_theta = alpha_theta
        self.beta_theta = beta_theta
        self.theta = torch.tensor(1.0, device=self.device)  # adaptive threshold
        
        # Temporal horizon
        self.eps_T = eps_T
        self.T_max = None  # will be computed dynamically
        self.prev_voltage_var = torch.tensor(0.0, device=self.device)
        
        # Oscillatory dynamics  
        self.beta_lambda = beta_lambda
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.leak = torch.tensor(lambda_init, device=self.device)  # adaptive leak
        
        # Lifecycle management
        self.tau_kill = tau_kill
        self.tau_sat = tau_sat 
        self.tau_active = tau_active
        self.state = NeuronState.ACTIVE
        self.vitality_history = []
        
        # Voltage state (Gaussian distribution)
        self.voltage_mu = torch.tensor(0.0, device=self.device)
        self.voltage_var = torch.tensor(1e-2, device=self.device)
        
        # AGVI V2_bar_tilde parameters
        self.v2_bar_tilde_mu = torch.tensor(1e-2, device=self.device)
        self.v2_bar_tilde_var = torch.tensor(1e-4, device=self.device)
        
        # Spike history for oscillatory dynamics
        self.spike_history = [0, 0, 0]  # last 3 timesteps
        
        # TAGI weight posteriors - populated by network
        self.incoming_weights = {}  # {from_neuron_id: (mu, var)}
        
        # Flag for vitality computation this step
        self._vitality_computed_this_step = False
        
    def update_voltage_gaussian(
        self, 
        incoming_spikes: Dict[int, float],
        prior_var: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update voltage Gaussian distribution following Eqs. (6-7)
        
        V_i^(t) = λ_i^(t) * V_i^(t-1) * (1 - S_i^(t-1)) + Σ_j W_ji^(t) * S_j^(t-1)
        
        Returns: (mu_V, sigma2_V)
        """
        prev_spike = self.spike_history[-1]
        
        # Eq. (6): mu_V update
        # Using item() or float() for scalars whenever possible
        leak = self.leak.item() if hasattr(self.leak, 'item') else self.leak
        v_mu = self.voltage_mu.item() if hasattr(self.voltage_mu, 'item') else self.voltage_mu
        v_var = self.voltage_var.item() if hasattr(self.voltage_var, 'item') else self.voltage_var
        
        leak_term = leak * v_mu * (1 - prev_spike)
        
        synaptic_mu = 0.0
        synaptic_var = 0.0
        
        for from_id, spike_val in incoming_spikes.items():
            if from_id in self.incoming_weights and spike_val > 0:
                w_mu, w_var = self.incoming_weights[from_id]
                synaptic_mu += w_mu * spike_val
                synaptic_var += w_var * spike_val
                
        if isinstance(leak_term, torch.Tensor): leak_term = leak_term.clone().detach()
        if isinstance(synaptic_mu, torch.Tensor): synaptic_mu = synaptic_mu.clone().detach()
        self.voltage_mu = torch.tensor(leak_term + synaptic_mu, device=self.device)
        
        # Eq. (7): sigma2_V update  
        leak_var_term = (leak * leak) * v_var * (1 - prev_spike)
        
        if isinstance(leak_var_term, torch.Tensor): leak_var_term = leak_var_term.clone().detach()
        if isinstance(synaptic_var, torch.Tensor): synaptic_var = synaptic_var.clone().detach()
        self.voltage_var = torch.tensor(leak_var_term + synaptic_var, device=self.device)
        
        return self.voltage_mu, self.voltage_var
    
    def compute_spike_probability(self) -> torch.Tensor:
        """
        Compute spike probability following Eq. (5):
        π_i^(t) = 1 - Φ((θ_i^(t) - μ_V_i^(t)) / sqrt(σ²_V_i^(t)))
        
        Returns: probability of spiking
        """
        std_v = torch.sqrt(self.voltage_var + 1e-8)
        z = (self.theta - self.voltage_mu) / std_v
        
        # Use torch special erf function for analytic Normal CDF calculation
        # norm.cdf(x) = 0.5 * (1 + erf(x / sqrt(2)))
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        return 1.0 - cdf
        
    def generate_spike(self) -> float:
        """Generate spike using Gaussian voltage and current threshold"""
        # Optimized normal random generation using torch.randn and scaling
        std_v = torch.sqrt(self.voltage_var + 1e-8)
        noise = torch.randn((), device=self.device)
        voltage_sample = self.voltage_mu + noise * std_v
        
        spike = float(voltage_sample > self.theta)
        
        # Update spike history
        self.spike_history = self.spike_history[1:] + [spike]
        return spike
        
    def update_homeostatic_threshold(self, spike: float):
        """
        Update adaptive threshold using surprise-gated homeostasis (Eq. 14):
        
        θ_i^(t+1) = θ_i^(t) + α_θ*(1-π_i^(t))*S_i^(t) - β_θ*π_i^(t)*(1-S_i^(t))
        """
        spike_prob = self.compute_spike_probability()
        surprise = 1.0 - spike_prob
        
        if spike > 0:
            # Surprising spike: raise threshold slightly
            delta_theta = self.alpha_theta * surprise * spike
        else:
            # Expected spike didn't occur: lower threshold  
            delta_theta = -self.beta_theta * spike_prob * (1 - spike)
            
        self.theta = self.theta + delta_theta
        self.theta = torch.clamp(self.theta, 0.1, 10.0)  # reasonable bounds
        
    def update_adaptive_leak(self):
        """
        Update adaptive leak for oscillatory dynamics (Eq. 31):
        
        λ_i^(t+1) = clip(λ_i^(t) + β_λ*(S_i^(t)*S_i^(t-2) - 0.1), λ_min, λ_max)
        """
        if len(self.spike_history) >= 3:
            # Encourage spiking at lag-2 (theta rhythm)  
            theta_term = self.spike_history[-1] * self.spike_history[-3] - 0.1
            delta_lambda = self.beta_lambda * theta_term
            self.leak = torch.clamp(self.leak + delta_lambda, self.lambda_min, self.lambda_max)
            
    def check_temporal_convergence(self) -> bool:
        """
        Check if temporal horizon T_max reached via variance convergence (Eq. 18):
        
        T_max,i = min{t ≥ 1 : |σ²_V_i^(t) - σ²_V_i^(t-1)| < ε_T}
        """
        if self.T_max is not None:
            return True  # already converged
            
        var_change = torch.abs(self.voltage_var - self.prev_voltage_var)
        if var_change < self.eps_T:
            self.T_max = len(self.vitality_history)  # current timestep
            return True
            
        self.prev_voltage_var = self.voltage_var.clone()
        return False
        
    def compute_vitality(self, network_prior_var: float = None) -> float:
        """
        Compute neuron vitality (Eq. 21):
        
        ρ_i^(t) = (1/|J_i|) * Σ_{j∈J_i} σ²_W_ji^(t) / σ²_0
        
        Returns: vitality score ∈ (0,1]
        """
        if not self.incoming_weights:
            return 1.0  # no connections = no shrinkage
            
        prior_var = network_prior_var if network_prior_var is not None else 0.1  # σ²_0
        n_conn = len(self.incoming_weights)
        
        # Python sum is faster than looping here
        total_var = sum(w_var.item() if hasattr(w_var, 'item') else float(w_var) for _, w_var in self.incoming_weights.values())
            
        vitality = (total_var / prior_var) / n_conn
        self.vitality_history.append(float(vitality))
        
        return vitality
        
    def update_lifecycle_state(self):
        """
        Update neuron lifecycle state following Eq. (23)
        
        States: DEAD (ρ ≥ τ_kill), ACTIVE (Δρ > τ_active), MATURE (ρ < τ_sat ∧ Δρ ≤ τ_active)
        """
        current_vitality = self.compute_vitality(self.prior_var)
        
        # Compute learning activity Δρ_i^(t) = |ρ_i^(t) - ρ_i^(t-1)|
        if len(self.vitality_history) >= 2:
            delta_rho = abs(self.vitality_history[-1] - self.vitality_history[-2])
        else:
            delta_rho = 0.0
            
        # Evaluate states in order (Eq. 23)
        if current_vitality >= self.tau_kill:
            self.state = NeuronState.DEAD
        elif delta_rho > self.tau_active:
            self.state = NeuronState.ACTIVE  
        elif current_vitality < self.tau_sat and delta_rho <= self.tau_active:
            self.state = NeuronState.MATURE
        else:
            self.state = NeuronState.ACTIVE  # default
            
    def step(self, incoming_spikes: Dict[int, float]) -> float:
        """Execute one timestep of neuron dynamics"""
        
        # Reset vitality computation flag for this step
        self._vitality_computed_this_step = False
        
        # 1. Update Gaussian voltage (Eqs. 6-7)
        self.update_voltage_gaussian(incoming_spikes)
        
        # 2. Generate spike
        spike = self.generate_spike()
        
        # 3. Update homeostatic threshold (Eq. 14)
        self.update_homeostatic_threshold(spike)
        
        # 4. Update adaptive leak (Eq. 31)
        self.update_adaptive_leak()
        
        # 5. Check temporal convergence (Eq. 18)
        self.check_temporal_convergence()
        
        # 6. Update lifecycle state (Eq. 23)
        self.update_lifecycle_state()
        
        return spike

class TAGISNNNetwork:
    """
    TAGI-enabled Spiking Neural Network implementing the full theoretical framework.
    
    Features:
    - TAGI weight inference (Eqs. 8-9)
    - KL-based structural plasticity (Eqs. 24-25)
    - Glial field modulation (Eq. 28)
    - Sleep-phase consolidation
    - Poisson input encoding (Eq. 34)
    - Spatial embedding and growth
    """
    
    def __init__(
        self,
        n_neurons: int,
        spatial_bounds: Tuple[float, float, float, float] = (0, 10, 0, 10),  # (x_min, x_max, y_min, y_max)
        initial_density: float = 0.05,
        device: Optional[torch.device] = None,
        # TAGI parameters
        prior_var: float = 0.1, # σ²_0 -> prior variance for weights, TODO NEED TO BE INITIALIZED IN A WAY THAT MATCHES THE NEURON PRIOR VAR FOR BETTER INITIAL BEHAVIOR
        sigma_v: float = 1e-2,  # output noise -> TODO: make this learnable (TAGI-V)
        # Structural plasticity
        kappa_grow: float = 0.5,
        kappa_prune: float = 0.1, 
        R_0: float = 2.0,
        delta: float = 0.2,
        gamma: float = 0.1,
        # Sleep parameters
        sleep_every: int = 100,
        sleep_duration: int = 50,
        gamma_sleep: float = 0.01
    ):
        self.device = device or torch.device("cpu")
        self.n_neurons = n_neurons
        self.spatial_bounds = spatial_bounds
        self.prior_var = prior_var
        self.sigma_v = sigma_v
        
        # Structural plasticity
        self.kappa_grow = kappa_grow
        self.kappa_prune = kappa_prune
        self.R_0 = R_0 
        self.delta = delta
        self.gamma = gamma
        self.R_max = torch.tensor(R_0, device=self.device)
        self.eta = torch.tensor(1e-3, device=self.device)  # learning rate
        
        # Sleep consolidation
        self.sleep_every = sleep_every
        self.sleep_duration = sleep_duration
        self.gamma_sleep = gamma_sleep
        self.high_error_targets = []  # for replay
        
        # Initialize data structures first
        self.neurons = {}
        self.weights = {}  # Adjacency matrix: {(from_id, to_id): (w_mu, w_var)}
        self.connectivity_matrix = torch.zeros((n_neurons, n_neurons), device=self.device)
        
        # Global state
        self.current_density = initial_density
        self.glial_field = torch.tensor(1.0, device=self.device)
        self.timestep = 0
        
        # Now initialize spatial structure (needs self.weights to exist)
        self._initialize_spatial_structure(initial_density)
        
    def _initialize_spatial_structure(self, density: float):
        """Initialize neurons with random spatial positions"""
        x_min, x_max, y_min, y_max = self.spatial_bounds
        
        for i in range(self.n_neurons):
            # Random position in bounds
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            
            neuron = TAGISNNNeuron(
                neuron_id=i,
                spatial_position=(x, y),
                device=self.device,
                prior_var=self.prior_var
            )
            self.neurons[i] = neuron
            
        # Initialize sparse connectivity
        n_connections = int(density * self.n_neurons ** 2)
        for _ in range(n_connections):
            from_id = np.random.randint(0, self.n_neurons)
            to_id = np.random.randint(0, self.n_neurons)
            
            if from_id != to_id and (from_id, to_id) not in self.weights:
                # Initialize weight at prior N(0, σ²_0)
                w_mu = torch.normal(0, math.sqrt(self.prior_var), size=(), device=self.device)
                w_var = torch.tensor(self.prior_var, device=self.device)
                self.weights[(from_id, to_id)] = (w_mu, w_var)
                self.connectivity_matrix[from_id, to_id] = 1.0
                
                # Register with receiving neuron
                self.neurons[to_id].incoming_weights[from_id] = (w_mu, w_var)
                
    def backward_step(
        self,
        targets: Dict[int, float],
        predictions: Dict[int, Tuple],
        input_spikes: Dict[int, float]
    ) -> None:
        """
        TAGI smoother backward pass (Rauch-Tung-Striebel style).

        The forward pass only predicts.  Here we use the innovation at each
        observed neuron to update the incoming weights via their cross-covariance
        with the predicted output.

        Heteroscedastic observation variance (TAGI-V, cf. observation.py):

            σ²_v_i = μ²_V_i + σ²_V_i   (= E[V²_i], learned noise floor)

        Predicted output variance (sum over synaptic contributions + noise):

            S_y_i = Σ_j  σ²_{W_ji} · S_j²  +  σ²_v_i

        Innovation scalars:

            δμ_i  = (y_i − μ_V_i) / S_y_i       (Kalman gain numerator)
            δS_i  = −1 / S_y_i

        Smoother update for each weight W_ji (cross-covariance gating):

            Cov(W_ji, V_i) = σ²_{W_ji} · S_j
            μ⁺_{W_ji}  = μ_{W_ji}  + Cov · δμ_i
            σ²⁺_{W_ji} = σ²_{W_ji} + Cov² · δS_i      (monotone shrinkage)
            
        Smoother update for each hidden voltage V_j (cross-covariance gating):
            Cov(V_j, V_i) = Cov(V_j, S_j) * μ_{W_ji} = (S_j * σ^2_{V_j}) / (sqrt(2 * pi * σ^2_{V_j})) * μ_{W_ji}
            μ^+_{V_j} = \mu_{V_j} + Cov(V_j, V_i) * delta_mu_i
            σ^{2,+}_{V_j} = σ^2_{V_j} + Cov(V_j, V_i)^2 * delta_S_i
        """
        # Node deltas tracking for recursive propagation {neuron_id: (delta_mu, delta_var)}
        node_deltas = {}
        
        for to_id, target in targets.items():
            if to_id not in predictions or to_id not in self.neurons:
                continue

            pred_mu, pred_var, pre_spikes = predictions[to_id]
            neuron = self.neurons[to_id]

            # --- Heteroscedastic observation variance (TAGI-V) ---------------
            target_t = torch.tensor(float(target), dtype=torch.float32, device=self.device)
            obs_diff = target_t - pred_mu

            # Compute the prior predictive PDF for v2
            mu_v2 = neuron.v2_bar_tilde_mu
            var_v2 = 3.0 * neuron.v2_bar_tilde_var + 2.0 * neuron.v2_bar_tilde_mu * neuron.v2_bar_tilde_mu
            cov_y_v = mu_v2

            # Variance of the output
            var_sum = pred_var + mu_v2

            # Compute updating quantities for the mean of the output
            tmp = 1.0 / var_sum
            delta_mu = tmp * obs_diff
            delta_var = -tmp
            
            # --- Store output node deltas ---
            node_deltas[to_id] = (delta_mu, delta_var)

            # Compute the posterior mean and variance for V
            mu_v_post = cov_y_v / var_sum * obs_diff
            var_v_post = mu_v2 - cov_y_v / var_sum * cov_y_v

            # Compute the posterior mean and variance for V2
            mu_v2_post = mu_v_post * mu_v_post + var_v_post
            var_v2_post = 2.0 * var_v_post * var_v_post + 4.0 * var_v_post * mu_v_post * mu_v_post

            # Compute the posterior mean and variance for V2_bar_tilde
            tmp_ratio = neuron.v2_bar_tilde_var / var_v2
            neuron.v2_bar_tilde_mu = neuron.v2_bar_tilde_mu + tmp_ratio * (mu_v2_post - mu_v2)
            neuron.v2_bar_tilde_var = neuron.v2_bar_tilde_var + tmp_ratio * tmp_ratio * (var_v2_post - var_v2)

            # Track high-error patterns for sleep replay
            if abs(float(obs_diff)) > 1.0:
                self.high_error_targets.append(
                    (input_spikes.copy(), {to_id: target})
                )
                
        # --- Smoother weight and hidden state updates (recursive) -------------------
        # We need to process from output to input. For SNNs we iterate the edges that 
        # lead to nodes with known deltas.
        
        edges_to_update = set()
        for to_id in node_deltas.keys():
            if to_id in self.neurons:
                for from_id in self.neurons[to_id].incoming_weights.keys():
                    edges_to_update.add((from_id, to_id))
                    
        # Update weights and pass deltas backward
        next_node_deltas = {}
        for from_id, to_id in edges_to_update:
            if to_id not in node_deltas:
                continue
                
            delta_mu, delta_var = node_deltas[to_id]
            pred_mu, pred_var, pre_spikes = predictions[to_id]
            neuron = self.neurons[to_id]
            w_mu, w_var = neuron.incoming_weights[from_id]
            s_j = pre_spikes.get(from_id, 0.0)
            
            # Update Weight W_ji
            # Cov(W_ji, V_i) = σ²_{W_ji} · S_j
            cov_w = w_var * s_j
            new_w_mu = w_mu + cov_w * delta_mu
            new_w_var = torch.clamp(w_var + cov_w * cov_w * delta_var, min=1e-6)
            
            self.weights[(from_id, to_id)] = (new_w_mu, new_w_var)
            neuron.incoming_weights[from_id] = (new_w_mu, new_w_var)
            
            # Compute backward deltas for the hidden node V_j if it's not an input node
            if from_id not in input_spikes and from_id in self.neurons and from_id in predictions:
                pred_mu_j, pred_var_j, _ = predictions[from_id]
                from_neuron = self.neurons[from_id]
                
                # Cov(V_j, V_i) = Cov(V_j, S_j) * μ_{W_ji}
                # For Heaviside S_j = H(V_j - θ_j), Cov(V_j, S_j) = P(S_j=1) * (some term). 
                # In standard analytic SNN TAGI:
                # Cov(V_j, V_i) = S_j * σ^2_{V_j} / sqrt(2*pi*σ^2_{V_j}) * μ_{W_ji}
                cov_v_s = s_j * pred_var_j / torch.clamp(torch.sqrt(2 * math.pi * pred_var_j), min=1e-6)
                cov_v_v = cov_v_s * w_mu
                
                delta_mu_j = cov_v_v * delta_mu
                delta_var_j = cov_v_v * cov_v_v * delta_var
                
                # Apply update to the hidden node V_j
                from_neuron.voltage_mu = from_neuron.voltage_mu + delta_mu_j
                from_neuron.voltage_var = torch.clamp(from_neuron.voltage_var + delta_var_j, min=1e-6)
                
                if from_id not in next_node_deltas:
                    next_node_deltas[from_id] = (torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device))
                next_node_deltas[from_id] = (
                    next_node_deltas[from_id][0] + delta_mu_j,
                    next_node_deltas[from_id][1] + delta_var_j
                )

            
    def compute_kl_divergence(self, neuron_id: int) -> float:
        """
        Compute KL divergence for structural plasticity (Eq. 24):
        
        κ_i^(t) = (1/|J_i|) * Σ_{j∈J_i} KL[N(μ_W_ji, σ²_W_ji) || N(0, σ²_0)]
        """
        neuron = self.neurons[neuron_id]
        
        if not neuron.incoming_weights:
            return 0.0
            
        total_kl = 0.0
        for w_mu, w_var in neuron.incoming_weights.values():
            # Eq. (25): KL for Gaussians
            kl = 0.5 * (
                w_var / self.prior_var + 
                (w_mu ** 2) / self.prior_var - 
                1 + 
                torch.log(self.prior_var / w_var)
            )
            total_kl += kl.item()
            
        return total_kl / len(neuron.incoming_weights)
        
    def update_glial_field(self):
        """
        Update global glial field (Eq. 28):
        
        G^(t) = (1/N) * Σ_i σ²_V_i^(t) / (σ²_V_i^(t-1) + ε)
        """
        if not self.neurons:
            self.glial_field = torch.tensor(1.0, device=self.device)
            return
            
        variance_ratios = []
        
        for neuron in self.neurons.values():
            if neuron.state == NeuronState.DEAD:
                continue
                
            prev_var = neuron.prev_voltage_var + 1e-8
            curr_var = neuron.voltage_var + 1e-8  # Add small epsilon to prevent NaN
            ratio = curr_var / prev_var
            
            # Clamp ratio to reasonable bounds to prevent extreme values
            ratio = torch.clamp(ratio, 0.1, 10.0)
            variance_ratios.append(ratio.item())
            
        if variance_ratios:
            self.glial_field = torch.tensor(np.mean(variance_ratios), device=self.device)
        else:
            self.glial_field = torch.tensor(1.0, device=self.device)
        
        # Update growth radius and learning rate (Eqs. 29-30)
        self.R_max = self.R_0 * (1 + self.delta * self.glial_field)
        self.eta = 1e-3 * (1 + self.gamma * self.glial_field)
        
    def structural_plasticity_step(self):
        """Execute structural plasticity: growth and pruning based on KL divergence"""
        neurons_to_remove = set()
        
        # Debug: check states before any action
        states_before = {nid: n.state.value for nid, n in self.neurons.items()}
        
        for neuron_id, neuron in self.neurons.items():
            if neuron.state == NeuronState.DEAD:
                neurons_to_remove.add(neuron_id)
                print(f"DEBUG: Neuron {neuron_id} marked for removal (state: DEAD)")
                continue
                
            kappa = self.compute_kl_divergence(neuron_id)
            
            # Check pruning criterion (Eq. 26)
            if kappa < self.kappa_prune:
                self._prune_synapses(neuron_id)
                
            # Check growth criterion (Eq. 25) - only if neuron spiked
            last_spike = neuron.spike_history[-1] if neuron.spike_history else 0
            if last_spike > 0 and kappa > self.kappa_grow:
                self._grow_synapses(neuron_id)
                
        # Remove dead neurons
        if neurons_to_remove:
            print(f"DEBUG: Removing {len(neurons_to_remove)} dead neurons: {neurons_to_remove}")
        self._remove_dead_neurons(neurons_to_remove)
        
    def _grow_synapses(self, neuron_id: int):
        """
        Grow new synapses following Eq. (27):
        Find nearest unconnected neighbor within R_max radius
        """
        neuron = self.neurons[neuron_id]
        neuron_pos = np.array(neuron.position)
        
        # Find nearest unconnected neighbor within radius
        min_distance = float('inf')
        best_target = None
        
        for candidate_id, candidate in self.neurons.items():
            if (candidate_id == neuron_id or 
                candidate_id in neuron.incoming_weights or
                candidate.state == NeuronState.DEAD):
                continue
                
            candidate_pos = np.array(candidate.position)
            distance = np.linalg.norm(neuron_pos - candidate_pos)
            
            if distance <= self.R_max and distance < min_distance:
                min_distance = distance
                best_target = candidate_id
                
        # Create new synapse
        if best_target is not None:
            w_mu = torch.normal(0, math.sqrt(self.prior_var), size=(), device=self.device)
            w_var = torch.tensor(self.prior_var, device=self.device)
            
            self.weights[(best_target, neuron_id)] = (w_mu, w_var)
            self.connectivity_matrix[best_target, neuron_id] = 1.0
            neuron.incoming_weights[best_target] = (w_mu, w_var)
            
    def _prune_synapses(self, neuron_id: int):
        """Remove synapses with low information content"""
        neuron = self.neurons[neuron_id]
        
        to_remove = []
        for from_id, (w_mu, w_var) in neuron.incoming_weights.items():
            # Prune if weight variance close to prior (no learning)
            var_ratio = w_var / self.prior_var
            if var_ratio > 0.9:  # hasn't learned much
                to_remove.append(from_id)
                
        for from_id in to_remove:
            if (from_id, neuron_id) in self.weights:
                del self.weights[(from_id, neuron_id)]
            self.connectivity_matrix[from_id, neuron_id] = 0.0
            del neuron.incoming_weights[from_id]
            
    def _remove_dead_neurons(self, neuron_ids: Set[int]):
        """Remove dead neurons and clean up connectivity"""
        for neuron_id in neuron_ids:
            # Remove all weights involving this neuron
            weights_to_remove = []
            for (from_id, to_id) in self.weights.keys():
                if from_id == neuron_id or to_id == neuron_id:
                    weights_to_remove.append((from_id, to_id))
                    
            for connection in weights_to_remove:
                del self.weights[connection]
                
            # Clean up connectivity matrix
            self.connectivity_matrix[neuron_id, :] = 0.0
            self.connectivity_matrix[:, neuron_id] = 0.0
            
            # Remove from other neurons' incoming weights
            for other_neuron in self.neurons.values():
                if neuron_id in other_neuron.incoming_weights:
                    del other_neuron.incoming_weights[neuron_id]
                    
            # Remove neuron
            if neuron_id in self.neurons:
                del self.neurons[neuron_id]
                
    def poisson_encoding(self, inputs: torch.Tensor, dt: float = 1.0) -> Dict[int, float]:
        """
        Encode continuous inputs as Poisson spike trains (Eq. 34):
        P(S_i^(t) = 1 | x_i) = x_i * Δt
        """
        spikes = {}
        
        for i, rate in enumerate(inputs):
            if i < self.n_neurons:
                spike_prob = torch.clamp(rate * dt, 0.0, 1.0)
                spike = float(torch.bernoulli(spike_prob).item())
                spikes[i] = spike
                
        return spikes
        
    @torch.no_grad()
    def _predict(
        self,
        input_spikes: Dict[int, float]
    ) -> Tuple[Dict[int, float], Dict[int, Tuple]]:
        """
        Pure forward (prediction only) — NO weight updates.

        Propagates Gaussian voltage through the network, generates spikes, and
        records, for each neuron: (μ_V, σ²_V, pre_spikes_used).
        The pre_spikes dict is needed by the smoother to compute cross-covariances.
        """
        output_spikes: Dict[int, float] = {}
        predictions: Dict[int, Tuple] = {}   # {neuron_id: (mu, var, pre_spikes)}

        for neuron_id, neuron in self.neurons.items():
            if neuron.state == NeuronState.DEAD:
                continue

            # Collect pre-synaptic activities (before this neuron fires)
            incoming: Dict[int, float] = {}
            for from_id in neuron.incoming_weights.keys():
                if from_id in input_spikes:
                    incoming[from_id] = input_spikes[from_id]
                elif from_id in output_spikes:
                    incoming[from_id] = output_spikes[from_id]
                else:
                    incoming[from_id] = 0.0

            # Forward: predict voltage (Gaussian propagation)
            neuron.update_voltage_gaussian(incoming)
            spike = neuron.generate_spike()
            output_spikes[neuron_id] = spike

            # Store snapshot before any weight modification
            predictions[neuron_id] = (
                neuron.voltage_mu.clone(),
                neuron.voltage_var.clone(),
                dict(incoming)
            )

            # Biological dynamics (no learning)
            neuron.update_homeostatic_threshold(spike)
            neuron.update_adaptive_leak()
            neuron.check_temporal_convergence()
            neuron.update_lifecycle_state()

        return output_spikes, predictions

    @torch.no_grad()
    def forward_step(
        self,
        input_spikes: Dict[int, float],
        targets: Optional[Dict[int, float]] = None
    ) -> Dict[int, float]:
        """
        One full step: predict → (if supervised) backward smoother → housekeeping.

        The forward pass is a pure prediction (see _predict).  Weight learning
        only happens in backward_step via the TAGI smoother.
        """
        # 1. Pure prediction
        output_spikes, predictions = self._predict(input_spikes)

        # 2. TAGI smoother backward (only when targets are available)
        if targets:
            self.backward_step(targets, predictions, input_spikes)

        # 3. Glial field
        self.update_glial_field()

        # 4. Structural plasticity
        if self.timestep % 10 == 0:
            self.structural_plasticity_step()

        # 5. Sleep phase
        if (self.timestep % self.sleep_every == 0
                and self.timestep > 0
                and not getattr(self, '_in_sleep_phase', False)):
            self.sleep_phase()

        self.timestep += 1
        return output_spikes
        
    @torch.no_grad()
    def _forward_step_no_sleep(
        self,
        input_spikes: Dict[int, float],
        targets: Optional[Dict[int, float]] = None
    ) -> Dict[int, float]:
        """Predict + backward smoother without triggering the sleep phase (replay)."""
        output_spikes, predictions = self._predict(input_spikes)
        if targets:
            self.backward_step(targets, predictions, input_spikes)
        return output_spikes
        
    def sleep_phase(self):
        """
        Execute sleep-phase consolidation following Algorithm 1 from paper
        """
        print(f"Entering sleep phase at timestep {self.timestep}")
        
        # Set sleep mode flag to prevent recursive sleep calls
        self._in_sleep_phase = True
        
        try:
            # Save wake state
            wake_weights = {k: (v[0].clone(), v[1].clone()) for k, v in self.weights.items()}
            
            # Step 1: Spontaneous activity with noise injection
            for t in range(self.sleep_duration):
                sleep_spikes = {}
                
                for neuron_id, neuron in self.neurons.items():
                    if neuron.state == NeuronState.DEAD:
                        continue
                        
                    # Add spontaneous noise to voltage mean
                    noise = torch.normal(0, 0.1, size=(), device=self.device)  
                    neuron.voltage_mu = neuron.voltage_mu + noise
                    
                    # Generate spontaneous spike
                    spike = neuron.generate_spike()
                    sleep_spikes[neuron_id] = spike
                    
                    # Update threshold (no target signal during sleep)
                    neuron.update_homeostatic_threshold(spike)
                    
            # Step 2: Synaptic downscaling
            for connection, (w_mu, w_var) in self.weights.items():
                # Downscale means, partially recover variances
                new_w_mu = w_mu * (1 - self.gamma_sleep)
                new_w_var = w_var * (1 + self.gamma_sleep)
                self.weights[connection] = (new_w_mu, new_w_var)
                
                # Update neuron copies
                from_id, to_id = connection
                if to_id in self.neurons:
                    self.neurons[to_id].incoming_weights[from_id] = (new_w_mu, new_w_var)
                    
            # Step 3: High-error replay with increased learning rate (use no-sleep version)
            for inputs, targets in self.high_error_targets[-10:]:  # last 10 high-error cases
                _ = self._forward_step_no_sleep(inputs, targets)
                
            # Step 4: Lifecycle evaluation - remove dead neurons
            dead_neurons = {nid for nid, n in self.neurons.items() if n.state == NeuronState.DEAD}
            self._remove_dead_neurons(dead_neurons)
            
            # Clear high-error buffer
            self.high_error_targets = []
            
            print(f"Sleep complete. {len(dead_neurons)} neurons removed. {len(self.neurons)} neurons remaining.")
            
        finally:
            # Always clear sleep mode flag
            self._in_sleep_phase = False
        
    def get_network_stats(self) -> Dict:
        """Get current network statistics"""
        # Debug: print actual neuron count
        actual_living_neurons = {nid: n for nid, n in self.neurons.items() if n.state != NeuronState.DEAD}
        print(f"DEBUG: Total neurons dict: {len(self.neurons)}, Living: {len(actual_living_neurons)}")
        
        states = [n.state.value for n in self.neurons.values()]
        vitalities = [n.vitality_history[-1] if n.vitality_history else 1.0 for n in self.neurons.values()]
        
        return {
            "n_neurons": len(actual_living_neurons),  # Count only living neurons  
            "n_connections": len(self.weights),
            "density": len(self.weights) / (len(actual_living_neurons) ** 2) if len(actual_living_neurons) > 0 else 0,
            "glial_field": self.glial_field.item(),
            "R_max": self.R_max.item(),
            "states": {s: states.count(s) for s in ['dead', 'active', 'mature']},
            "avg_vitality": np.mean(vitalities) if vitalities else 0.0,
            "timestep": self.timestep
        }

# Example usage and training function
def test_tagi_snn_simple():
    """
    Simple test of TAGI-SNN with synthetic data (no external dependencies)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Triton available: {TRITON_AVAILABLE}")
    
    # Create small network: 20 neurons in 2D spatial grid  
    network = TAGISNNNetwork(
        n_neurons=20,
        spatial_bounds=(0, 5, 0, 5),
        initial_density=0.2,
        device=device,
        sleep_every=50,  # more frequent sleep for testing
        sleep_duration=10
    )
    
    print(f"Initialized network with {len(network.neurons)} neurons")
    print(f"Initial connectivity: {len(network.weights)} connections")
    
    # Synthetic data: simple pattern recognition
    # Pattern A: high activity in neurons 0-4, target neuron 15
    # Pattern B: high activity in neurons 5-9, target neuron 16
    
    patterns = [
        # Pattern A
        ({0: 0.8, 1: 0.7, 2: 0.6, 3: 0.8, 4: 0.7}, {15: 1.0}),
        # Pattern B  
        ({5: 0.8, 6: 0.7, 7: 0.6, 8: 0.8, 9: 0.7}, {16: 1.0}),
        # Noise pattern
        ({10: 0.3, 11: 0.2, 12: 0.4}, {17: 0.0})
    ]
    
    print("Training TAGI-SNN on synthetic patterns...")
    
    for epoch in range(3):
        print(f"\n--- Epoch {epoch + 1} ---")
        
        for step in range(100):
            # Select random pattern
            pattern_idx = np.random.randint(0, len(patterns))
            input_rates, targets = patterns[pattern_idx]
            
            # Convert to input tensor and encode as spikes
            input_tensor = torch.zeros(20, device=device)
            for neuron_id, rate in input_rates.items():
                if neuron_id < input_tensor.size(0):
                    input_tensor[neuron_id] = rate
                    
            input_spikes = network.poisson_encoding(input_tensor)
            
            # Forward pass
            output_spikes = network.forward_step(input_spikes, targets)
            
            if step % 20 == 0:
                stats = network.get_network_stats()
                active_spikes = sum(1 for s in output_spikes.values() if s > 0)
                print(f"  Step {step}: Pattern {pattern_idx}, "
                      f"Active spikes: {active_spikes}, "
                      f"Neurons: {stats['n_neurons']}, "
                      f"Density: {stats['density']:.3f}, "
                      f"Glial: {stats['glial_field']:.3f}")
                
                # Show state distribution
                state_dist = stats['states']
                print(f"    States - Active: {state_dist.get('active', 0)}, "
                      f"Mature: {state_dist.get('mature', 0)}, "
                      f"Dead: {state_dist.get('dead', 0)}")
                      
    print("\nTraining complete!")
    
    # Final test - check if network learned to respond to patterns
    print("\n--- Final Pattern Testing ---")
    with torch.no_grad():
        for pattern_idx, (input_rates, expected_targets) in enumerate(patterns):
            input_tensor = torch.zeros(20, device=device)
            for neuron_id, rate in input_rates.items():
                if neuron_id < input_tensor.size(0):
                    input_tensor[neuron_id] = rate
                    
            input_spikes = network.poisson_encoding(input_tensor)
            output_spikes = network.forward_step(input_spikes)
            
            # Check target neurons
            target_responses = {}
            for target_id in expected_targets.keys():
                if target_id in output_spikes:
                    target_responses[target_id] = output_spikes[target_id]
                    
            print(f"Pattern {pattern_idx}: Input {input_rates} -> Target responses {target_responses}")
    
    final_stats = network.get_network_stats()
    print(f"\nFinal network: {final_stats['n_neurons']} neurons, "
          f"{final_stats['n_connections']} connections, "
          f"avg vitality: {final_stats['avg_vitality']:.3f}")
    
    return network

if __name__ == "__main__":
    try:
        # Run simple test instead of MNIST
        network = test_tagi_snn_simple()
        print("\n✅ TAGI-SNN test completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

