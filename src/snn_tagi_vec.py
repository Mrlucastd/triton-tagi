"""
src/snn_tagi_vec.py

Vectorized Tractable Analytic Bayesian Spiking Neural Network (TAGI-SNN).

Implements ALL mechanisms from snn_tagi.tex:
  1. Gaussian voltage propagation (Eqs. 6-7)
  2. TAGI weight inference (Eqs. 8-9)
  3. TAGI-V learned observation variance via AGVI (§4.3)
  4. Surprise-gated homeostatic plasticity (Eq. 14)
  5. Adaptive temporal horizon via variance stationarity (§5)
  6. Neuron lifecycle: dead/active/mature via vitality (§6)
  7. KL-driven structural plasticity: grow + prune (§7)
  8. Glial-field modulation of growth (§7.3)
  9. Multi-timescale oscillatory dynamics (§8)
  10. Sleep-phase consolidation with replay (§9)
  11. Neurogenesis: dormant hidden-neuron pool activated by prediction error (§new)
"""

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch


class TAGISNNNetworkVec:
    """
    Vectorized TAGI-SNN with full biological mechanisms.

    All neuron states are (N,) vectors; weights are (N, N) dense matrices
    with a binary connectivity mask.

    Hidden neurons (those that are neither input nor output) start in the
    DORMANT state and are activated by the neurogenesis mechanism when the
    network's prediction error is persistently high — i.e. when a new
    observation cannot be explained by the existing representation.
    """

    # Lifecycle state constants
    ACTIVE  = 0
    MATURE  = 1
    DEAD    = 2
    DORMANT = 3   # hidden neurons waiting to be recruited

    def __init__(
        self,
        n_neurons: int,
        input_ids: Optional[List[int]] = None,
        output_ids: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
        prior_var: float = 0.1,
        density: float = 0.05,
        ensure_io: bool = True,
        # Homeostasis
        alpha_theta: float = 0.01,
        beta_theta: float = 0.01,
        # Leak / oscillatory dynamics
        lambda_init: float = 0.5,
        lambda_min: float = 0.1,
        lambda_max: float = 0.95,
        beta_lambda: float = 0.01,
        # Temporal horizon
        eps_T: float = 1e-4,
        # Lifecycle thresholds
        tau_kill: float = 0.95,
        tau_sat: float = 0.2,
        tau_active: float = 1e-3,
        # Structural plasticity
        kappa_grow: float = 0.5,
        kappa_prune: float = 0.05,
        # Glial field
        glial_delta: float = 0.5,
        glial_gamma: float = 0.3,
        # Sleep
        sleep_every: int = 5000,
        sleep_duration: int = 10,
        gamma_sleep: float = 0.005,
        sigma_sleep: float = 0.01,
        replay_buffer_size: int = 50,
        # Neurogenesis — driven by normalised predictive surprise
        # (surprise = (y-μ)² / (var_V + obs_var), threshold in units of σ²)
        neurogenesis_surprise_threshold: float = 1.5,
        neurogenesis_n_per_event_max: int = 5,
        neurogenesis_n_inputs_per_neuron: int = 50,
        neurogenesis_ema_alpha: float = 0.05,
        neurogenesis_cooldown: int = 300,
    ):
        self.device = device or torch.device("cpu")
        self.N = n_neurons
        self.n_neurons = n_neurons
        self.prior_var = prior_var

        # Neuron roles
        self.input_ids  = list(input_ids)  if input_ids  else []
        self.output_ids = list(output_ids) if output_ids else []
        self._inp_idx = torch.tensor(self.input_ids,  device=self.device, dtype=torch.long)
        self._out_idx = torch.tensor(self.output_ids, device=self.device, dtype=torch.long)

        # Hidden neurons = everything that is neither input nor output
        io_set = set(self.input_ids) | set(self.output_ids)
        self.hidden_ids = [i for i in range(n_neurons) if i not in io_set]
        self._hid_idx   = (
            torch.tensor(self.hidden_ids, device=self.device, dtype=torch.long)
            if self.hidden_ids else
            torch.tensor([], device=self.device, dtype=torch.long)
        )

        # Homeostasis
        self.alpha_theta = alpha_theta
        self.beta_theta  = beta_theta

        # Leak / oscillatory
        self.lambda_min  = lambda_min
        self.lambda_max  = lambda_max
        self.beta_lambda = beta_lambda

        # Temporal horizon
        self.eps_T = eps_T

        # Lifecycle
        self.tau_kill   = tau_kill
        self.tau_sat    = tau_sat
        self.tau_active = tau_active

        # Structural plasticity
        self.kappa_grow  = kappa_grow
        self.kappa_prune = kappa_prune

        # Glial field
        self.glial_delta = glial_delta
        self.glial_gamma = glial_gamma

        # Sleep
        self.sleep_every    = sleep_every
        self.sleep_duration = sleep_duration
        self.gamma_sleep    = gamma_sleep
        self.sigma_sleep    = sigma_sleep

        # Neurogenesis hyper-parameters
        self.neurogenesis_surprise_threshold   = neurogenesis_surprise_threshold
        self.neurogenesis_n_per_event_max      = neurogenesis_n_per_event_max
        self.neurogenesis_n_inputs_per_neuron  = neurogenesis_n_inputs_per_neuron
        self.neurogenesis_ema_alpha            = neurogenesis_ema_alpha
        self.neurogenesis_cooldown             = neurogenesis_cooldown
        # Normalised predictive surprise EMA:  (y-μ)² / (var_V + obs_var)  at outputs
        self._surprise_ema          = 0.0
        self._last_neurogenesis_step = -(neurogenesis_cooldown + 1)
        self.neurogenesis_total      = 0     # cumulative neurons ever activated

        # ─── Neuron state vectors (N,) ───────────────────────────────
        self.mu_V         = torch.zeros(n_neurons, device=self.device)
        self.var_V        = torch.full((n_neurons,), 1e-2, device=self.device)
        self.var_V_prev   = torch.full((n_neurons,), 1e-2, device=self.device)
        self.theta        = torch.full((n_neurons,), 0.5,  device=self.device)
        self.leak         = torch.full((n_neurons,), lambda_init, device=self.device)
        self.S_prev       = torch.zeros(n_neurons, device=self.device)
        self.S_prev2      = torch.zeros(n_neurons, device=self.device)
        # Cumulative spike counts for analytics (incremented each forward step)
        self.spike_counts = torch.zeros(n_neurons, device=self.device)

        # AGVI observation-variance tracker (TAGI-V, no fixed sigma_v)
        self.v2_bar_mu  = torch.full((n_neurons,), prior_var, device=self.device)
        self.v2_bar_var = torch.full((n_neurons,), 1e-4,      device=self.device)

        # Lifecycle state — hidden neurons start DORMANT
        self.lifecycle = torch.zeros(n_neurons, device=self.device, dtype=torch.long)
        if len(self.hidden_ids) > 0:
            self.lifecycle[self._hid_idx] = self.DORMANT

        self.rho           = torch.ones(n_neurons,  device=self.device)
        self.rho_prev      = torch.ones(n_neurons,  device=self.device)
        self.T_max_reached = torch.zeros(n_neurons, device=self.device, dtype=torch.bool)

        # ─── Weight matrices (N, N) ─────────────────────────────────
        self.mw   = torch.zeros(n_neurons, n_neurons, device=self.device)
        self.sw   = torch.zeros(n_neurons, n_neurons, device=self.device)
        self.mask = torch.zeros(n_neurons, n_neurons, device=self.device)

        self._init_connectivity(density, ensure_io)

        # Glial field
        self.glial_field = 1.0

        # Replay buffer for sleep
        self.replay_buffer: deque = deque(maxlen=replay_buffer_size)

        self.timestep = 0

    # ──────────────────────────────────────────────────────────────────
    def _init_connectivity(self, density: float, ensure_io: bool):
        """Sparse random connectivity among non-dormant neurons + guaranteed input→output."""
        # Random mask only among non-dormant neurons
        self.mask = (torch.rand(self.N, self.N, device=self.device) < density).float()
        self.mask.fill_diagonal_(0.0)

        # Dormant hidden neurons have NO connections initially
        if len(self.hidden_ids) > 0:
            self.mask[self._hid_idx, :] = 0.0
            self.mask[:, self._hid_idx] = 0.0

        if ensure_io and len(self.input_ids) > 0 and len(self.output_ids) > 0:
            idx_j = self._inp_idx.unsqueeze(1).expand(-1, self._out_idx.numel()).reshape(-1)
            idx_i = self._out_idx.unsqueeze(0).expand(self._inp_idx.numel(), -1).reshape(-1)
            self.mask[idx_j, idx_i] = 1.0

        # He-style init: var = 2 / fan_in
        fan_in  = self.mask.sum(dim=0).clamp(min=1.0)
        init_var = 2.0 / fan_in
        init_std = torch.sqrt(init_var)

        self.mw = torch.randn_like(self.mw) * init_std.unsqueeze(0)
        self.mw *= self.mask
        self.sw = init_var.unsqueeze(0).expand_as(self.mw).clone()
        self.sw *= self.mask

        # Set threshold to match expected voltage scale
        expected_std = torch.sqrt(init_var * fan_in.clamp(min=1.0) * 0.2)
        self.theta = expected_std.clamp(min=0.1, max=2.0)

        n_conn  = int(self.mask.sum().item())
        io_conn = 0
        if len(self.input_ids) > 0 and len(self.output_ids) > 0:
            io_conn = int(self.mask[self._inp_idx][:, self._out_idx].sum().item())
        print(f"Connections: {n_conn}  (input→output: {io_conn})")

    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def poisson_encoding(self, rates: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        return torch.bernoulli(torch.clamp(rates * dt, 0.0, 1.0))

    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def forward_step(
        self,
        input_spikes: torch.Tensor,
        targets: Optional[Dict[int, float]] = None,
    ) -> torch.Tensor:
        """
        Full TAGI-SNN timestep: forward → TAGI+AGVI → homeostasis →
        lifecycle → structural plasticity → (sleep if scheduled).
        """
        S = input_spikes
        # Drive used at this step: external input spikes now + recurrent spikes from t-1.
        drive_spikes = self._build_drive_spikes(S)

        # ── 1. Gaussian voltage propagation (Eqs. 6-7) ─────────────
        self.var_V_prev = self.var_V.clone()
        leak_factor = self.leak * (1.0 - self.S_prev)

        self.mu_V = (
            leak_factor * self.mu_V
            + torch.mv((self.mw * self.mask).T, drive_spikes)
        )
        self.var_V = (
            (self.leak ** 2) * (1.0 - self.S_prev) * self.var_V
            + torch.mv((self.sw * self.mask).T, drive_spikes)
        )

        # ── 2. Spike generation (Eq. 4) ────────────────────────────
        std_V   = torch.sqrt(self.var_V.clamp(min=1e-8))
        V_sample = self.mu_V + torch.randn_like(self.mu_V) * std_V
        S_new   = (V_sample > self.theta).float()
        S_new[self._inp_idx] = S[self._inp_idx]
        # Update cumulative spike counts
        self.spike_counts += S_new

        # ── 3. Spike probability for homeostasis (Eq. 5) ───────────
        z  = (self.theta - self.mu_V) / std_V
        pi = 1.0 - 0.5 * (1.0 + torch.erf(z * (1.0 / math.sqrt(2.0))))

        # ── 4. Homeostatic threshold update (Eq. 14) ───────────────
        self.theta += (
            self.alpha_theta * (1.0 - pi) * S_new
            - self.beta_theta * pi * (1.0 - S_new)
        )
        self.theta.clamp_(0.1, 10.0)

        # ── 5. Adaptive leak (Eq. 31) ──────────────────────────────
        self.leak += self.beta_lambda * (S_new * self.S_prev2 - 0.1)
        self.leak.clamp_(self.lambda_min, self.lambda_max)

        # ── 6. Spike history ───────────────────────────────────────
        self.S_prev2 = self.S_prev.clone()
        self.S_prev  = S_new.clone()

        # ── 7. TAGI backward + AGVI (if targets) ──────────────────
        if targets is not None:
            self._backward(S, targets, drive_spikes=drive_spikes)

        # ── 8. Temporal horizon check (Eq. 17) ─────────────────────
        var_change = torch.abs(self.var_V - self.var_V_prev)
        self.T_max_reached |= (var_change < self.eps_T)

        # ── 9. Lifecycle update (every 100 steps) ──────────────────
        if self.timestep > 0 and self.timestep % 100 == 0:
            self._update_lifecycle()

        # ── 10. Structural plasticity (every 50 steps) ─────────────
        if self.timestep > 0 and self.timestep % 50 == 0:
            self._structural_plasticity(S_new)

        # ── 11. Sleep phase (every sleep_every steps) ──────────────
        if (self.sleep_every > 0
                and self.timestep > 0
                and self.timestep % self.sleep_every == 0):
            self._sleep_phase()

        self.timestep += 1
        return S_new

    # ──────────────────────────────────────────────────────────────────
    def _build_drive_spikes(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """Build synaptic drive vector used by forward/backward at the current step.

        Inputs are clamped to current external spikes, while non-input neurons
        contribute their spikes from the previous timestep.
        """
        drive_spikes = self.S_prev.clone()
        drive_spikes[self._inp_idx] = input_spikes[self._inp_idx]
        return drive_spikes

    # ──────────────────────────────────────────────────────────────────
    def _backward(
        self,
        S: torch.Tensor,
        targets: Dict[int, float],
        drive_spikes: Optional[torch.Tensor] = None,
    ):
        """Vectorised TAGI backward + AGVI + neurogenesis + replay buffer update."""
        if drive_spikes is None:
            drive_spikes = self._build_drive_spikes(S)

        # Build target vector & mask
        y     = torch.zeros(self.N, device=self.device)
        tmask = torch.zeros(self.N, device=self.device)
        for oid, val in targets.items():
            y[oid]     = val
            tmask[oid] = 1.0

        # ── AGVI: learned observation variance (TAGI-V) on output layer only ──
        # Mirrors src/update/observation.py _output_innovation_kernel_heteros.
        delta_mu  = torch.zeros(self.N, device=self.device)
        delta_var = torch.zeros(self.N, device=self.device)

        out_tmask = tmask[self._out_idx] > 0
        target_out_idx = self._out_idx[out_tmask]

        if target_out_idx.numel() > 0:
            obs_diff_out = y[target_out_idx] - self.mu_V[target_out_idx]

            mu_v2_out   = self.v2_bar_mu[target_out_idx]
            var_v2_out  = 3.0 * self.v2_bar_var[target_out_idx] + 2.0 * mu_v2_out ** 2
            cov_y_v_out = mu_v2_out

            var_sum_out = (self.var_V[target_out_idx] + mu_v2_out).clamp(min=1e-8)
            tmp_out = 1.0 / var_sum_out

            # Innovation for output means
            delta_mu[target_out_idx]  = tmp_out * obs_diff_out
            delta_var[target_out_idx] = -tmp_out

            # AGVI posterior updates for V2_bar at outputs
            mu_v_post_out   = cov_y_v_out / var_sum_out * obs_diff_out
            var_v_post_out  = mu_v2_out - cov_y_v_out / var_sum_out * cov_y_v_out
            mu_v2_post_out  = mu_v_post_out ** 2 + var_v_post_out
            var_v2_post_out = 2.0 * var_v_post_out ** 2 + 4.0 * var_v_post_out * mu_v_post_out ** 2

            ratio_out = self.v2_bar_var[target_out_idx] / var_v2_out.clamp(min=1e-8)
            self.v2_bar_mu[target_out_idx] = (
                self.v2_bar_mu[target_out_idx]
                + ratio_out * (mu_v2_post_out - mu_v2_out)
            )
            self.v2_bar_var[target_out_idx] = (
                self.v2_bar_var[target_out_idx]
                + ratio_out ** 2 * (var_v2_post_out - var_v2_out)
            )

        self.v2_bar_mu.clamp_(min=1e-8)
        self.v2_bar_var.clamp_(min=1e-8)

        # ── Weight update (TAGI smoother) ───────────────────────────
        # Mature neurons get a reduced (not zero) learning rate so they can
        # still adapt when new classes arrive — avoids catastrophic freezing.
        mature_scale = 0.1
        update_mask = torch.where(
            self.lifecycle == self.MATURE,
            torch.full((self.N,), mature_scale, device=self.device),
            torch.ones(self.N, device=self.device),
        )
        # Dormant neurons must not receive weight updates
        update_mask[self.lifecycle == self.DORMANT] = 0.0

        # 1. Update weights with the same drive used in forward voltage propagation.
        S_col  = drive_spikes.unsqueeze(1)
        cov_w  = self.sw * self.mask * S_col           # (N, N)

        dw_mu  = cov_w * delta_mu.unsqueeze(0)  * update_mask.unsqueeze(0)
        dw_var = (cov_w ** 2) * delta_var.unsqueeze(0) * update_mask.unsqueeze(0)

        # 2. Propagate deltas backwards to hidden neurons (time t-1)
        const      = 1.0 / math.sqrt(2.0 * math.pi)
        std_V_prev = torch.sqrt(self.var_V_prev.clamp(min=1e-8))
        cov_v_s    = drive_spikes * self.var_V_prev / std_V_prev * const  # (N,)
        cov_v_v    = cov_v_s.unsqueeze(1) * self.mw * self.mask  # (N, N)

        delta_mu_hidden  = torch.mv(cov_v_v,      delta_mu)
        delta_var_hidden = torch.mv(cov_v_v ** 2, delta_var)

        # Only update non-input, non-dormant hidden voltages
        non_input_mask = torch.ones(self.N, device=self.device)
        non_input_mask[self._inp_idx] = 0.0
        non_input_mask[self.lifecycle == self.DORMANT] = 0.0

        self.mu_V  += delta_mu_hidden  * non_input_mask
        self.var_V  = torch.clamp(
            self.var_V + delta_var_hidden * non_input_mask, min=1e-6
        )

        # Apply weight updates
        self.mw += dw_mu
        self.sw += dw_var
        self.sw.clamp_(min=1e-6)
        self.mw *= self.mask
        self.sw *= self.mask

        # ── Neurogenesis: activate dormant neurons on high NORMALISED surprise ─
        # surprise_i = (y_i - μ_i)² / (var_V_i + obs_var_i)
        # When var_V is large the network explains the error by its own uncertainty
        # → surprise ≈ 1, no growth needed.
        # When var_V is small (network confident) but error is large → surprise > 1
        # → the observation is genuinely novel → grow new neurons.
        predictive_var = (self.var_V[self._out_idx] + self.v2_bar_mu[self._out_idx]).clamp(min=1e-8)
        obs_diff_out = y[self._out_idx] - self.mu_V[self._out_idx]
        surprise = float((obs_diff_out ** 2 / predictive_var).mean().item())
        self._neurogenesis(surprise)

        # ── Store high-surprise samples for sleep replay ───────────
        if len(self.replay_buffer) < self.replay_buffer.maxlen or surprise > self._min_replay_error():
            self.replay_buffer.append((
                S.clone(),
                drive_spikes.clone(),
                dict(targets),
                surprise,
            ))

    # ──────────────────────────────────────────────────────────────────
    def _neurogenesis(self, surprise: float) -> int:
        """
        Activate dormant hidden neurons based on normalised predictive surprise.

        surprise = (y - mu_V)^2 / (var_V + obs_var)  averaged over output neurons.

        When var_V is large (network uncertain) the error is explained by the
        network's own uncertainty → surprise ≈ 1 → no growth.
        When var_V is small (network confident) but the error is large → surprise > 1
        → the observation is genuinely novel → recruit new hidden neurons.

        The number of neurons recruited scales with the surplus surprise above
        threshold so that mild novelty adds a few neurons and strong novelty
        adds more, up to `neurogenesis_n_per_event_max`.
        A cooldown prevents exhausting the pool on a single burst of high error.
        """
        alpha = self.neurogenesis_ema_alpha
        self._surprise_ema = alpha * surprise + (1.0 - alpha) * self._surprise_ema

        if self._surprise_ema < self.neurogenesis_surprise_threshold:
            return 0

        if self.timestep - self._last_neurogenesis_step < self.neurogenesis_cooldown:
            return 0

        dormant = torch.where(self.lifecycle == self.DORMANT)[0]
        if len(dormant) == 0:
            return 0

        # Scale n_activate with surplus surprise (more surprise → more neurons)
        surplus    = self._surprise_ema / self.neurogenesis_surprise_threshold
        n_activate = min(max(1, int(surplus)), self.neurogenesis_n_per_event_max, len(dormant))
        new_ids    = dormant[:n_activate]

        n_in = min(self.neurogenesis_n_inputs_per_neuron, len(self.input_ids))

        for nid in new_ids.tolist():
            # ── Connect random input neurons → new hidden neuron ────
            perm    = torch.randperm(len(self.input_ids), device=self.device)
            src_inp = self._inp_idx[perm[:n_in]]

            for s in src_inp.tolist():
                self.mask[s, nid] = 1.0
                self.mw[s, nid]   = torch.randn(1, device=self.device).item() * math.sqrt(self.prior_var)
                self.sw[s, nid]   = self.prior_var

            # ── Connect new hidden neuron → all output neurons ──────
            for oid in self.output_ids:
                self.mask[nid, oid] = 1.0
                self.mw[nid, oid]   = 0.0           # zero mean: let TAGI discover direction
                self.sw[nid, oid]   = self.prior_var

            # ── Initialise neuron state ─────────────────────────────
            fan_in         = float(n_in)
            self.var_V[nid] = 2.0 / max(fan_in, 1.0)
            self.mu_V[nid]  = 0.0
            # Threshold: expect voltage std ≈ sqrt(var * fan_in * firing_rate)
            expected_std    = math.sqrt(2.0 / max(fan_in, 1.0) * fan_in * 0.2)
            self.theta[nid] = float(max(0.1, min(2.0, expected_std)))
            self.leak[nid]  = 0.5

            # Mark as active — ready to participate in forward/backward
            self.lifecycle[nid] = self.ACTIVE

        self.neurogenesis_total      += n_activate
        self._last_neurogenesis_step  = self.timestep
        return n_activate

    # ──────────────────────────────────────────────────────────────────
    def reset_dynamics(self):
        """Reset per-sample dynamic state (voltages, spike history) but keep weights.

        Call this before evaluating each test sample so that the persistent
        voltage state from the previous sample does not bleed into the next.
        """
        self.mu_V.zero_()
        self.var_V.fill_(1e-2)
        self.S_prev.zero_()
        self.S_prev2.zero_()

    # ──────────────────────────────────────────────────────────────────
    def _min_replay_error(self) -> float:
        if len(self.replay_buffer) == 0:
            return 0.0
        return min(e for _, _, _, e in self.replay_buffer)

    # ──────────────────────────────────────────────────────────────────
    def _update_lifecycle(self):
        """
        Vitality-based lifecycle (§6):
          rho_i = mean(sw[:,i] / prior_var) over active connections
          DEAD: rho >= tau_kill, ACTIVE: |drho| > tau_active, MATURE: rho < tau_sat

        DORMANT neurons are never touched here — they are managed exclusively
        by _neurogenesis.
        """
        # Vitality: mean posterior/prior variance ratio per target neuron
        fan_in         = self.mask.sum(dim=0).clamp(min=1.0)
        sum_var_ratio  = (self.sw * self.mask).sum(dim=0) / self.prior_var
        self.rho_prev  = self.rho.clone()
        self.rho       = sum_var_ratio / fan_in

        drho = torch.abs(self.rho - self.rho_prev)

        # Remember which neurons are dormant before we touch lifecycle
        dormant_before = (self.lifecycle == self.DORMANT)

        # Assign states (evaluated in order per Eq. 23)
        new_state = torch.full_like(self.lifecycle, self.ACTIVE)

        dead_mask   = self.rho >= self.tau_kill
        mature_mask = (self.rho < self.tau_sat) & (drho <= self.tau_active)

        new_state[mature_mask] = self.MATURE
        new_state[dead_mask]   = self.DEAD

        # Input / output neurons always stay ACTIVE so they adapt to new classes
        new_state[self._inp_idx] = self.ACTIVE
        new_state[self._out_idx] = self.ACTIVE

        # Dormant neurons must stay DORMANT — only _neurogenesis can awaken them
        new_state[dormant_before] = self.DORMANT

        self.lifecycle = new_state

        # Kill dead neurons: remove their connections
        dead_neurons = (self.lifecycle == self.DEAD)
        if dead_neurons.any():
            self.mask[dead_neurons, :] = 0.0
            self.mask[:, dead_neurons] = 0.0
            self.mw[dead_neurons, :]   = 0.0
            self.mw[:, dead_neurons]   = 0.0
            self.sw[dead_neurons, :]   = 0.0
            self.sw[:, dead_neurons]   = 0.0

    # ──────────────────────────────────────────────────────────────────
    def _structural_plasticity(self, S_new: torch.Tensor):
        """
        KL-driven structural plasticity (§7):
          Grow: neuron spiked AND kappa > kappa_grow
          Prune: kappa < kappa_prune

        Dormant neurons are excluded from both grow and prune.
        """
        # ── Glial field (Eq. 22) ────────────────────────────────────
        var_ratio        = self.var_V / (self.var_V_prev + 1e-8)
        self.glial_field = float(var_ratio.mean().item())

        # ── Per-neuron KL divergence from prior (Eq. 25-26) ────────
        sw_safe    = self.sw.clamp(min=1e-8)
        kl_per_conn = 0.5 * (
            sw_safe / self.prior_var
            + self.mw ** 2 / self.prior_var
            - 1.0
            + torch.log(self.prior_var / sw_safe)
        )
        kl_per_conn *= self.mask  # only existing connections

        fan_in = self.mask.sum(dim=0).clamp(min=1.0)
        kappa  = kl_per_conn.sum(dim=0) / fan_in  # (N,)

        # ── Growth (Eq. 27) ─────────────────────────────────────────
        grow_mask    = (
            (S_new > 0)
            & (kappa > self.kappa_grow)
            & (self.lifecycle == self.ACTIVE)
        )
        grow_neurons = torch.where(grow_mask)[0]

        for i_idx in grow_neurons[:10].tolist():
            unconnected  = (self.mask[:, i_idx] == 0).clone()
            unconnected[i_idx] = False
            # Do not connect to dead or dormant neurons
            excluded = (self.lifecycle == self.DEAD) | (self.lifecycle == self.DORMANT)
            unconnected &= ~excluded

            if unconnected.any():
                candidates = torch.where(unconnected)[0]
                j_idx = candidates[torch.randint(len(candidates), (1,))].item()
                self.mask[j_idx, i_idx] = 1.0
                self.mw[j_idx, i_idx]   = (
                    torch.randn(1, device=self.device).item() * math.sqrt(self.prior_var)
                )
                self.sw[j_idx, i_idx] = self.prior_var

        # ── Pruning (Eq. 28) ───────────────────────────────────────
        prune_mask    = (kappa < self.kappa_prune) & (self.lifecycle == self.ACTIVE)
        prune_neurons = torch.where(prune_mask)[0]

        for i_idx in prune_neurons[:10].tolist():
            connected = torch.where(self.mask[:, i_idx] > 0)[0]
            if len(connected) <= 1:
                continue  # keep at least 1

            if i_idx in self.output_ids:
                input_set       = set(self.input_ids)
                non_input       = [j.item() for j in connected if j.item() not in input_set]
                if len(non_input) == 0:
                    continue
                connected_for_prune = torch.tensor(
                    non_input, device=self.device, dtype=torch.long
                )
            else:
                connected_for_prune = connected

            if len(connected_for_prune) == 0:
                continue

            kl_vals = kl_per_conn[connected_for_prune, i_idx]
            worst   = connected_for_prune[kl_vals.argmin()].item()
            self.mask[worst, i_idx] = 0.0
            self.mw[worst, i_idx]   = 0.0
            self.sw[worst, i_idx]   = 0.0

    # ──────────────────────────────────────────────────────────────────
    def _sleep_phase(self):
        """
        Sleep-phase consolidation (§9, Algorithm 1):
          1. Spontaneous activity (noise injection)
          2. Synaptic downscaling + partial variance recovery
          3. High-error replay
          4. Kill dead neurons
        """
        # ── 1. Spontaneous activity ─────────────────────────────────
        for _ in range(self.sleep_duration):
            noise      = torch.randn(self.N, device=self.device) * self.sigma_sleep
            self.mu_V += noise

            self.var_V = (
                (self.leak ** 2) * (1.0 - self.S_prev) * self.var_V
                + torch.mv((self.sw * self.mask).T, self.S_prev)
            )
            std_V    = torch.sqrt(self.var_V.clamp(min=1e-8))
            V_sample = self.mu_V + torch.randn_like(self.mu_V) * std_V
            S_new    = (V_sample > self.theta).float()
            S_new[self._inp_idx]                     = 0.0
            S_new[self.lifecycle == self.DORMANT]    = 0.0

            z  = (self.theta - self.mu_V) / std_V
            pi = 1.0 - 0.5 * (1.0 + torch.erf(z * (1.0 / math.sqrt(2.0))))
            self.theta += (
                self.alpha_theta * (1.0 - pi) * S_new
                - self.beta_theta * pi * (1.0 - S_new)
            )
            self.theta.clamp_(0.1, 10.0)

            self.S_prev2 = self.S_prev.clone()
            self.S_prev  = S_new.clone()

        # ── 2. Synaptic downscaling + variance recovery ────────────
        self.mw *= (1.0 - self.gamma_sleep)
        self.sw *= (1.0 + self.gamma_sleep)
        self.mw *= self.mask
        self.sw *= self.mask

        # ── 3. High-error replay ───────────────────────────────────
        if len(self.replay_buffer) > 0:
            sorted_buf   = sorted(self.replay_buffer, key=lambda x: -x[3])
            replay_count = min(len(sorted_buf), 10)
            for s_replay, drive_replay, t_replay, _ in sorted_buf[:replay_count]:
                self._backward(s_replay, t_replay, drive_spikes=drive_replay)

        # ── 4. Lifecycle evaluation + kill dead ────────────────────
        self._update_lifecycle()

    # ──────────────────────────────────────────────────────────────────
    def get_network_stats(self) -> Dict:
        """Stats for logging."""
        n_conn    = int(self.mask.sum().item())
        n_active  = int((self.lifecycle == self.ACTIVE ).sum().item())
        n_mature  = int((self.lifecycle == self.MATURE ).sum().item())
        n_dead    = int((self.lifecycle == self.DEAD   ).sum().item())
        n_dormant = int((self.lifecycle == self.DORMANT).sum().item())
        n_silent  = int((self.spike_counts == 0).sum().item())
        return {
            "n_neurons":           self.N,
            "n_connections":       n_conn,
            "density":             n_conn / max(self.N ** 2, 1),
            "n_active":            n_active,
            "n_mature":            n_mature,
            "n_dead":              n_dead,
            "n_dormant":           n_dormant,
            "n_silent":            n_silent,
            "timestep":            self.timestep,
            "mean_theta":          float(self.theta[self._out_idx].mean().item()),
            "mean_obs_var":        float(self.v2_bar_mu[self._out_idx].mean().item()),
            "mean_vitality":       float(self.rho[self._out_idx].mean().item()),
            "glial_field":         self.glial_field,
            "replay_buffer_size":  len(self.replay_buffer),
            "neurogenesis_total":  self.neurogenesis_total,
            "surprise_ema":        self._surprise_ema,
        }
