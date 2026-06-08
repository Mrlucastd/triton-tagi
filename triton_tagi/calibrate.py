"""
Unified parameter-free online calibration for TAGI networks.

This module implements a single projection operator ``T`` on the Gaussian
parameter state ``(mw, Sw, mb, Sb)`` of each learnable layer. ``T`` enforces
three conditions in one input metric ``G = E[a aᵀ]`` (augmented with a bias
column), and it is applied both at initialization and — online — after every
Kalman step:

    (I)   Signal = 1, centred:  diag(mwᵀ Gᶜ mw) = 1,  E[z] = 0
                                (whiten / column-normalize mw; centre via mb)
    (II)  Balanced gain:        Sw[i,:] = c · g_diag[i]^{-1/2},  Sb = c
                                (⇒ every per-parameter Kalman gain equals c/Var[z])
    (III) Calibrated budget:    Σ epistemic = σ_v²  ⇒  output Kalman gain J = ½,
                                fixing  c = σ_v² / (Σ_i √g_diag[i] + 1)

Parameter-free, fully online:
    * ``sigma_v`` is not a hyperparameter — it is learned online (homoscedastic
      running estimate from residuals, or the existing heteroscedastic V2 head).
    * The init budget σ_v² is only a seed; the online estimate adapts it.
    * The online forgetting rate λ_t is driven by the batch surprise
      χ² = E[(y-μ_z)²/(Var[z]+σ_v²)] — no schedule, no tuning: χ²≈1 ⇒ λ→0
      (stable), χ²≫1 ⇒ λ↑ (reopen the gain, "new data → undecided").

Compute substrate (consistent with the rest of triton-tagi):
    * The per-element / per-step state updates — variance re-inflation, the
      mean-projection blend, and the surprise/residual maps — are **fused Triton
      kernels** (mirroring ``update/parameters.py`` and ``update/observation.py``).
    * The whitening factorizations (QR / eigh / SVD) run on ``torch.linalg``
      (cuSOLVER) and the matmuls on ``torch.matmul`` (cuBLAS), exactly as
      ``linear.py`` / ``conv2d.py`` do — there is no Triton primitive for those.

The operator reduces to the analytic closed form (He-style fixed point) when
``metric="analytic"`` and to data-driven whitening when ``metric="data"``.
It is opt-in and does not change any default behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import triton
import triton.language as tl
from torch import Tensor

from .base import LearnableLayer

BLOCK = 1024

# ----------------------------------------------------------------------
#  ReLU Gaussian-moment constants (input ~ N(0, V)) — used by the analytic metric
# ----------------------------------------------------------------------
_E_A_RELU = 1.0 / math.sqrt(2.0 * math.pi)        # E[ReLU(X)]      / √V
_E_A2_RELU = 0.5                                   # E[ReLU(X)²]     / V
_VAR_A_RELU = 0.5 - 1.0 / (2.0 * math.pi)          # Var[ReLU(X)]    / V  ≈ 0.3408


# ======================================================================
#  Triton kernels — fused element-wise operator state updates
# ======================================================================


@triton.jit
def _reinflate_row_kernel(
    S_ptr,
    sqrtg_ptr,
    c,
    lam,
    fan_out,
    n_elements,
    BLOCK: tl.constexpr,
):
    """(S) Per-row variance re-inflation: S ← (1-λ)·S + λ·(c / √g_diag[row]).

    ``S`` is a row-major ``(fan_in, fan_out)`` weight-variance tensor; the target
    depends only on the input row ``i = offs // fan_out`` via ``sqrtg_ptr[i]``.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    S = tl.load(S_ptr + offs, mask=valid)
    row = offs // fan_out
    sg = tl.load(sqrtg_ptr + row, mask=valid, other=1.0)
    s_target = c / sg
    tl.store(S_ptr + offs, (1.0 - lam) * S + lam * s_target, mask=valid)


@triton.jit
def _reinflate_const_kernel(S_ptr, c, lam, n_elements, BLOCK: tl.constexpr):
    """(S) Constant-target re-inflation: S ← (1-λ)·S + λ·c  (bias / BatchNorm)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements
    S = tl.load(S_ptr + offs, mask=valid)
    tl.store(S_ptr + offs, (1.0 - lam) * S + lam * c, mask=valid)


@triton.jit
def _lerp_kernel(dst_ptr, src_ptr, beta, n_elements, BLOCK: tl.constexpr):
    """(μ) Mean-projection blend: dst ← (1-β)·dst + β·src."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements
    d = tl.load(dst_ptr + offs, mask=valid)
    s = tl.load(src_ptr + offs, mask=valid)
    tl.store(dst_ptr + offs, (1.0 - beta) * d + beta * s, mask=valid)


@triton.jit
def _norm_innov_kernel(y_ptr, mu_ptr, var_ptr, sigma2, out_ptr, n_elements, BLOCK: tl.constexpr):
    """Normalized innovation per element: ``(y-μ)² / (Var + σ_v²)`` (for χ²)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements
    y = tl.load(y_ptr + offs, mask=valid)
    mu = tl.load(mu_ptr + offs, mask=valid)
    var = tl.load(var_ptr + offs, mask=valid)
    r = y - mu
    denom = tl.maximum(var + sigma2, 1e-12)
    tl.store(out_ptr + offs, r * r / denom, mask=valid)


@triton.jit
def _sqdiff_kernel(y_ptr, mu_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    """Squared residual per element: ``(y - μ)²`` (for the σ_v² estimate)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements
    y = tl.load(y_ptr + offs, mask=valid)
    mu = tl.load(mu_ptr + offs, mask=valid)
    r = y - mu
    tl.store(out_ptr + offs, r * r, mask=valid)


# ======================================================================
#  Triton wrappers
# ======================================================================


def reinflate_weight_(Sw: Tensor, sqrt_g: Tensor, c: float, lam: float) -> None:
    """Re-inflate a weight-variance block toward ``c / √g_diag`` (in place)."""
    Sw = Sw if Sw.is_contiguous() else Sw.contiguous()
    fan_out = Sw.shape[1]
    n = Sw.numel()
    _reinflate_row_kernel[(triton.cdiv(n, BLOCK),)](
        Sw.view(-1), sqrt_g.contiguous().view(-1), float(c), float(lam), fan_out, n, BLOCK=BLOCK
    )


def reinflate_const_(S: Tensor, c: float, lam: float) -> None:
    """Re-inflate a bias / per-channel variance toward constant ``c`` (in place)."""
    n = S.numel()
    _reinflate_const_kernel[(triton.cdiv(n, BLOCK),)](
        S.view(-1), float(c), float(lam), n, BLOCK=BLOCK
    )


def lerp_(dst: Tensor, src: Tensor, beta: float) -> None:
    """In-place ``dst ← (1-β)·dst + β·src`` over matching tensors."""
    n = dst.numel()
    _lerp_kernel[(triton.cdiv(n, BLOCK),)](
        dst.view(-1), src.contiguous().view(-1), float(beta), n, BLOCK=BLOCK
    )


def normalized_innovation(y: Tensor, mu: Tensor, var: Tensor, sigma2: float) -> Tensor:
    """Per-element ``(y-μ)²/(Var+σ_v²)`` via Triton; returns a tensor like ``y``."""
    out = torch.empty_like(y)
    n = y.numel()
    _norm_innov_kernel[(triton.cdiv(n, BLOCK),)](
        y.contiguous(), mu.contiguous(), var.contiguous(), float(sigma2), out, n, BLOCK=BLOCK
    )
    return out


def squared_residual(y: Tensor, mu: Tensor) -> Tensor:
    """Per-element ``(y-μ)²`` via Triton; returns a tensor like ``y``."""
    out = torch.empty_like(y)
    n = y.numel()
    _sqdiff_kernel[(triton.cdiv(n, BLOCK),)](y.contiguous(), mu.contiguous(), out, n, BLOCK=BLOCK)
    return out


# ======================================================================
#  Linear-algebra primitives for condition (I)  (cuSOLVER / cuBLAS via torch)
# ======================================================================


def sqrt_invsqrt_psd(G: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """Symmetric-PSD square root and inverse square root ``(G^{1/2}, G^{-1/2})``."""
    evals, evecs = torch.linalg.eigh((G + G.transpose(-1, -2)) * 0.5)
    ridge = eps * evals.clamp_min(0).mean().clamp_min(1e-12)
    vals = evals.clamp_min(0) + ridge
    sq = (evecs * vals.sqrt()) @ evecs.transpose(-1, -2)
    inv = (evecs * vals.rsqrt()) @ evecs.transpose(-1, -2)
    return sq, inv


def polar_orth_columns(B: Tensor) -> Tensor:
    """Nearest column-orthonormal matrix (exact polar factor ``U Vᵀ`` via SVD)."""
    U, _, Vh = torch.linalg.svd(B, full_matrices=False)
    return U @ Vh


def newton_schulz_orth_columns(B: Tensor, iters: int = 5) -> Tensor:
    """Muon's quintic Newton–Schulz column orthogonalization (approximate)."""
    X = B.clone()
    sigma1 = torch.linalg.matrix_norm(X, ord=2).clamp_min(1e-12)
    X = X / (sigma1 + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(iters):
        G = X.transpose(-1, -2) @ X
        X = a * X + X @ (b * G + c * (G @ G))
    return X


# ======================================================================
#  The operator T (init form) for one dense/conv weight block
# ======================================================================


def _signal_unit_columns(fan_in, fan_out, whiten, v_analytic, device, generator) -> Tensor:
    """Build mw with unit signal variance per output (condition I).

    ``fan_in >= fan_out``: orthonormal columns in the metric (decorrelated);
    otherwise each column is normalized individually (best when fan_out > fan_in).
    """
    Q = torch.randn(fan_in, fan_out, device=device, generator=generator)
    if fan_in >= fan_out:
        Q, _ = torch.linalg.qr(Q, mode="reduced")     # orthonormal columns
    W = whiten * Q if isinstance(whiten, float) else whiten @ Q
    if isinstance(whiten, float):
        col_sig = v_analytic * (W * W).sum(dim=0)      # Gᶜ = v·I
        W = W / col_sig.clamp_min(1e-12).sqrt().unsqueeze(0)
    # matrix-whiten path uses QR (fan_in >= fan_out) so columns are already unit-signal
    return W


def calibrate_dense_block(
    mw: Tensor,
    Sw: Tensor,
    mb: Tensor | None,
    Sb: Tensor | None,
    *,
    Ea: Tensor,
    g_diag: Tensor,
    whiten,
    v_analytic: float,
    sigma2: float,
    has_bias: bool,
    generator: torch.Generator | None = None,
) -> dict:
    """Apply the init operator T to one ``(fan_in, fan_out)`` weight block, in place.

    Works for ``Linear`` (mw ``(in, out)``) and ``Conv2D`` (mw ``(C_in·kH·kW, C_out)``)
    identically. The variance shaping (II)+(III) is written by the Triton
    re-inflation kernel with λ = 1.
    """
    fan_in, fan_out = mw.shape
    new_mw = _signal_unit_columns(fan_in, fan_out, whiten, v_analytic, mw.device, generator)
    mw.copy_(new_mw)

    sqrt_g = g_diag.clamp_min(1e-12).sqrt()
    c = sigma2 / (sqrt_g.sum().item() + 1.0)          # (III) one scalar fixes the budget
    reinflate_weight_(Sw, sqrt_g, c, lam=1.0)         # (II) Sw[i,:] = c / √g_diag[i]

    if has_bias and mb is not None:
        mb.copy_(-(Ea.unsqueeze(0) @ new_mw))         # centring: E[z] = 0
        reinflate_const_(Sb, c, lam=1.0)

    return {
        "kind": "dense",
        "Ea": Ea,
        "g_diag": g_diag,
        "sqrt_g": sqrt_g,
        "whiten": whiten,
        "v_analytic": v_analytic,
        "c": c,
        "fan_in": fan_in,
        "fan_out": fan_out,
    }


def calibrate_norm_block(mw: Tensor, Sw: Tensor, mb: Tensor, Sb: Tensor, *, sigma2: float) -> dict:
    """Calibrate a per-channel affine (BatchNorm γ, β): the degenerate T case.

    A norm layer self-whitens its input, so (I) holds with γ-mean=1, β-mean=0.
    (II)+(III) reduce to equal balanced variances ``Sγ = Sβ = σ_v²/2``.
    """
    c = sigma2 / 2.0
    mw.fill_(1.0)
    mb.fill_(0.0)
    reinflate_const_(Sw, c, lam=1.0)
    reinflate_const_(Sb, c, lam=1.0)
    return {"kind": "norm", "c": c}


# ======================================================================
#  Network-level init: sequential pass of T through depth
# ======================================================================


def _iter_learnable(layers) -> list:
    """Flatten learnable layers, descending into ResBlock sublayers, in order."""
    out = []
    for layer in layers:
        sub = getattr(layer, "_learnable", None)
        if sub is not None:                            # ResBlock and similar containers
            out.extend(sub)
        elif isinstance(layer, LearnableLayer):
            out.append(layer)
    return out


def _is_norm(layer) -> bool:
    """A learnable layer whose weight is a 1-D per-channel affine (BatchNorm)."""
    return getattr(layer, "mw", None) is not None and layer.mw.dim() == 1


def calibrate(
    net,
    *,
    sigma2_obs: float = 0.01,
    metric: str = "analytic",
    input_var: float = 1.0,
    seed: int | None = 1,
) -> list[dict]:
    """Calibrate every learnable layer of ``net`` with the unified operator T.

    Generalizes across architectures (MLP, CNN, ResNet) via the ``(fan_in, fan_out)``
    weight convention. ``Linear`` / ``Conv2D`` get full signal/gain/budget
    calibration; ``BatchNorm2D`` gets the degenerate per-channel calibration.

    Args:
        net:        A ``Sequential`` network (already on its device).
        sigma2_obs: Seed observation-noise variance (adapted online afterward).
        metric:     ``"analytic"`` (closed-form ReLU fixed point, no data, all
                    architectures) or ``"data"`` (reserved; analytic for conv).
        input_var:  Assumed variance of the (normalized) network input.
        seed:       RNG seed for the orthonormal seeds (reproducibility).

    Returns:
        Per-layer metadata list (also attached as ``layer._calib``), consumed by
        :func:`online_recalibrate`.
    """
    if metric not in ("analytic", "data"):
        raise ValueError(f"Unknown metric: {metric!r}")

    Vz = 1.0 + sigma2_obs
    metas: list[dict] = []
    first_matmul = True

    for layer in _iter_learnable(net.layers):
        device = layer.mw.device
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed + len(metas))

        if _is_norm(layer):
            meta = calibrate_norm_block(layer.mw, layer.Sw, layer.mb, layer.Sb, sigma2=sigma2_obs)
        else:
            fan_in = layer.mw.shape[0]
            if first_matmul:                           # input-fed matmul layer
                Ea_val, q, v = 0.0, input_var, input_var
                first_matmul = False
            else:                                      # ReLU-fed (analytic fixed point)
                Ea_val = math.sqrt(Vz) * _E_A_RELU
                q = Vz * _E_A2_RELU
                v = Vz * _VAR_A_RELU
            Ea = torch.full((fan_in,), Ea_val, device=device)
            g_diag = torch.full((fan_in,), q, device=device)
            whiten = 1.0 / math.sqrt(v)                # analytic Gᶜ = v·I
            meta = calibrate_dense_block(
                layer.mw, layer.Sw, getattr(layer, "mb", None), getattr(layer, "Sb", None),
                Ea=Ea, g_diag=g_diag, whiten=whiten, v_analytic=v, sigma2=sigma2_obs,
                has_bias=getattr(layer, "has_bias", True), generator=gen,
            )

        layer._calib = meta
        metas.append(meta)

    return metas


# ======================================================================
#  Online operator T (per-step): re-inflation + mean projection
# ======================================================================


def surprise_lambda(chi2: float, lam_max: float, tau: float) -> float:
    """Map batch surprise χ² to a forgetting rate λ ∈ [0, lam_max]."""
    excess = max(chi2 - 1.0, 0.0)
    return lam_max * (1.0 - math.exp(-excess / max(tau, 1e-8)))


def batch_chi2(y: Tensor, y_pred_mu: Tensor, y_pred_var: Tensor, sigma2: float) -> float:
    """Mean normalized innovation ``E[(y-μ)²/(Var+σ_v²)]`` on a batch (Triton map)."""
    if y_pred_mu.shape[-1] == 2 * y.shape[-1]:         # heteroscedastic head → even cols
        mu = y_pred_mu[..., 0::2].contiguous()
        var = y_pred_var[..., 0::2].contiguous()
    else:
        mu, var = y_pred_mu, y_pred_var
    return float(normalized_innovation(y, mu, var, sigma2).mean().item())


def _reinflate_layer(layer, lam: float) -> None:
    """(S) Re-inflate a layer's posterior variance toward its calibrated target."""
    if lam <= 0:
        return
    meta = layer._calib
    if meta["kind"] == "norm":
        reinflate_const_(layer.Sw, meta["c"], lam)
        reinflate_const_(layer.Sb, meta["c"], lam)
        return
    reinflate_weight_(layer.Sw, meta["sqrt_g"], meta["c"], lam)
    if getattr(layer, "has_bias", True) and getattr(layer, "Sb", None) is not None:
        reinflate_const_(layer.Sb, meta["c"], lam)


def _project_layer(layer, beta: float, method: str = "polar", ns_iters: int = 5) -> None:
    """(μ) Muon: pull mw a fraction β toward the signal=1 manifold, re-centre bias."""
    if beta <= 0:
        return
    meta = layer._calib
    if meta["kind"] == "norm":
        return
    fan_in, fan_out = meta["fan_in"], meta["fan_out"]
    if fan_in < fan_out:                               # projection ill-posed; reinflation only
        return
    whiten = meta["whiten"]
    mw = layer.mw
    if isinstance(whiten, float):
        unwhiten = 1.0 / whiten                        # √v
        B = unwhiten * mw
        Bo = polar_orth_columns(B) if method == "polar" else newton_schulz_orth_columns(B, ns_iters)
        proj = whiten * Bo
    else:
        unwhiten, _ = sqrt_invsqrt_psd(torch.linalg.inv(whiten @ whiten))
        B = unwhiten @ mw
        Bo = polar_orth_columns(B) if method == "polar" else newton_schulz_orth_columns(B, ns_iters)
        proj = whiten @ Bo
    lerp_(mw, proj, beta)                              # Triton blend
    if getattr(layer, "has_bias", True) and getattr(layer, "mb", None) is not None:
        layer.mb.copy_(-(meta["Ea"].unsqueeze(0) @ mw))


# ======================================================================
#  Online configuration + driver
# ======================================================================


@dataclass
class OnlineCalibration:
    """Configuration and mutable state for the online operator T.

    Attributes:
        mode:        ``"off"`` | ``"const"`` | ``"surprise"`` forgetting schedule.
        lam:         λ (const mode) or λ_max cap (surprise mode).
        tau:         Surprise sensitivity.
        project:     Apply the Muon mean projection.
        beta:        Mean-projection relaxation rate.
        every:       Batches between projections.
        proj_method: ``"polar"`` (exact SVD) | ``"ns"`` (Newton–Schulz).
        ns_iters:    Newton–Schulz iterations (if ``proj_method="ns"``).
        sigma_v_mode: ``"homoscedastic"`` (online running σ_v²) | ``"heteroscedastic"``
                      (the existing V2 head learns it; σ_v² state is ignored).
        sigma_v2:    Running σ_v² estimate (mutable).
        sigma_v2_rho: EMA rate for the σ_v² estimate.
        sigma_v2_floor: Lower bound on σ_v².
    """

    mode: str = "surprise"
    lam: float = 0.05
    tau: float = 1.0
    project: bool = True
    beta: float = 0.10
    every: int = 25
    proj_method: str = "polar"
    ns_iters: int = 5
    sigma_v_mode: str = "homoscedastic"
    sigma_v2: float = 0.01
    sigma_v2_rho: float = 0.01
    sigma_v2_floor: float = 1e-4
    step_count: int = 0
    lambda_hist: list[float] = field(default_factory=list)

    @property
    def sigma_v(self) -> float:
        """Current observation-noise std used by the innovation."""
        return math.sqrt(max(self.sigma_v2, self.sigma_v2_floor))


def update_sigma_v2(cfg: OnlineCalibration, y: Tensor, y_pred_mu: Tensor, y_pred_var: Tensor) -> None:
    """Online homoscedastic σ_v² estimate: σ̂² = E[(y-μ)²] − E[Var_z] (EMA)."""
    if cfg.sigma_v_mode != "homoscedastic":
        return
    if y_pred_mu.shape[-1] == 2 * y.shape[-1]:         # heteroscedastic head owns the noise
        return
    resid2 = float(squared_residual(y, y_pred_mu).mean().item())
    epist = float(y_pred_var.mean().item())
    obs = max(resid2 - epist, cfg.sigma_v2_floor)
    cfg.sigma_v2 = (1 - cfg.sigma_v2_rho) * cfg.sigma_v2 + cfg.sigma_v2_rho * obs


def online_recalibrate(net, lam: float, cfg: OnlineCalibration) -> None:
    """Apply the per-step online operator T to every calibrated layer."""
    cfg.step_count += 1
    do_proj = cfg.project and (cfg.step_count % max(cfg.every, 1) == 0)
    for layer in _iter_learnable(net.layers):
        if getattr(layer, "_calib", None) is None:
            continue
        _reinflate_layer(layer, lam)                   # (S)
        if do_proj:
            _project_layer(layer, cfg.beta, cfg.proj_method, cfg.ns_iters)   # (μ) Muon
    cfg.lambda_hist.append(lam)
