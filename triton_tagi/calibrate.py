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
    * ``sigma_v`` is not a hyperparameter. The Bayesian, TAGI-native way to learn
      it is TAGI-V / AGVI — the latent error-variance head (``sigma_v_mode=
      "heteroscedastic"``; see ``examples/regression_heteros.py`` and
      ``_output_innovation_kernel_heteros``). A method-of-moments running estimate
      (``"homoscedastic"``) is offered as a lightweight non-AGVI fallback.
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

The operator reduces to the analytic closed form (He-style ReLU fixed point) when
``metric="analytic"`` (``Gᶜ = v·I``, all architectures) and to data-driven
whitening when ``metric="data"`` (``Gᶜ`` = measured Gram of real activations,
dense networks). It is opt-in and does not change any default behaviour.
"""

from __future__ import annotations

import math
from collections import deque
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

# Relative ridge added to the empirical Gram before its inverse-sqrt (data metric).
_REG_EPS = 1e-4


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
    """Re-inflate a weight-variance block toward ``c / √g_diag`` (in place).

    Non-contiguous ``Sw`` is updated through a contiguous scratch buffer that is
    copied back, so the operation is genuinely in place for any layout.
    """
    buf = Sw if Sw.is_contiguous() else Sw.contiguous()
    fan_out = buf.shape[1]
    n = buf.numel()
    _reinflate_row_kernel[(triton.cdiv(n, BLOCK),)](
        buf.view(-1), sqrt_g.contiguous().view(-1), float(c), float(lam), fan_out, n, BLOCK=BLOCK
    )
    if buf is not Sw:
        Sw.copy_(buf)


def reinflate_const_(S: Tensor, c: float, lam: float) -> None:
    """Re-inflate a bias / per-channel variance toward constant ``c`` (in place).

    Handles non-contiguous ``S`` by updating a contiguous copy and writing it back.
    """
    buf = S if S.is_contiguous() else S.contiguous()
    n = buf.numel()
    _reinflate_const_kernel[(triton.cdiv(n, BLOCK),)](
        buf.view(-1), float(c), float(lam), n, BLOCK=BLOCK
    )
    if buf is not S:
        S.copy_(buf)


def lerp_(dst: Tensor, src: Tensor, beta: float) -> None:
    """In-place ``dst ← (1-β)·dst + β·src`` over matching tensors.

    Handles non-contiguous ``dst`` by updating a contiguous copy and writing it back.
    """
    buf = dst if dst.is_contiguous() else dst.contiguous()
    n = buf.numel()
    _lerp_kernel[(triton.cdiv(n, BLOCK),)](
        buf.view(-1), src.contiguous().view(-1), float(beta), n, BLOCK=BLOCK
    )
    if buf is not dst:
        dst.copy_(buf)


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
    """Nearest column-orthonormal matrix: the EXACT polar factor ``U Vᵀ`` via SVD.

    This is the only orthogonalization used by the Muon mean projection —
    approximations (e.g. Newton–Schulz iterations) are intentionally not provided,
    so ``signal = 1`` and exact output decorrelation hold after every projection.
    """
    U, _, Vh = torch.linalg.svd(B, full_matrices=False)
    return U @ Vh


# ======================================================================
#  The operator T (init form) for one dense/conv weight block
# ======================================================================


def _signal_unit_columns(fan_in, fan_out, whiten, v_analytic, device, generator) -> Tensor:
    """Build mw with unit signal variance per output (condition I).

    The seed ``Q`` has unit-norm columns (orthonormal when ``fan_in >= fan_out``,
    individually normalized otherwise). With a matrix whiten ``W_c = (Gᶜ)^{-1/2}``,
    ``mw = W_c Q`` then has signal ``diag(mwᵀ Gᶜ mw) = diag(Qᵀ Q) = 1`` exactly,
    since ``(Gᶜ)^{1/2} W_c = I``. With a scalar (analytic) whiten the columns are
    rescaled explicitly by their realized signal ``v · Σ_i mw[i,o]²``.
    """
    Q = torch.randn(fan_in, fan_out, device=device, generator=generator)
    if fan_in >= fan_out:
        Q, _ = torch.linalg.qr(Q, mode="reduced")          # orthonormal (unit-norm) columns
    else:
        Q = Q / Q.norm(dim=0, keepdim=True).clamp_min(1e-12)   # unit-norm columns
    if isinstance(whiten, float):
        W = whiten * Q
        col_sig = v_analytic * (W * W).sum(dim=0)          # Gᶜ = v·I
        W = W / col_sig.clamp_min(1e-12).sqrt().unsqueeze(0)
    else:
        W = whiten @ Q                                     # signal_o = ‖Q[:,o]‖² = 1
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
    unwhiten=None,
    generator: torch.Generator | None = None,
) -> dict:
    """Apply the init operator T to one ``(fan_in, fan_out)`` weight block, in place.

    Works for ``Linear`` (mw ``(in, out)``) and ``Conv2D`` (mw ``(C_in·kH·kW, C_out)``)
    identically. The variance shaping (II)+(III) is written by the Triton
    re-inflation kernel with λ = 1.

    ``whiten`` / ``unwhiten`` are ``(Gᶜ)^{-1/2}`` / ``(Gᶜ)^{1/2}`` — scalars in the
    analytic metric (``Gᶜ = v·I``) or matrices in the data metric. ``unwhiten`` is
    stored for the online mean projection; if omitted it defaults to ``1/whiten``
    (valid only for the scalar case).
    """
    if unwhiten is None:
        unwhiten = 1.0 / whiten                       # scalar analytic fallback (√v)
    fan_in, fan_out = mw.shape
    new_mw = _signal_unit_columns(fan_in, fan_out, whiten, v_analytic, mw.device, generator)
    mw.copy_(new_mw)

    sqrt_g = g_diag.clamp_min(1e-12).sqrt()
    budget_denom = sqrt_g.sum().item() + 1.0          # Σ_i √g_diag[i] + 1  (bias column)
    c = sigma2 / budget_denom                         # (III) one scalar fixes the budget
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
        "unwhiten": unwhiten,
        "v_analytic": v_analytic,
        "c": c,
        "budget_denom": budget_denom,                  # c = σ_v²/budget_denom (online tracking)
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
    return {"kind": "norm", "c": c, "budget_denom": 2.0}


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


def _make_generator(seed: int | None, device, offset: int) -> torch.Generator | None:
    """Per-layer RNG (seed + offset) so calibration is reproducible across runs."""
    if seed is None:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + offset)
    return gen


def calibrate(
    net,
    *,
    sigma2_obs: float = 0.01,
    metric: str = "analytic",
    input_var: float = 1.0,
    seed: int | None = 1,
    data_batch: Tensor | None = None,
) -> list[dict]:
    """Calibrate every learnable layer of ``net`` with the unified operator T.

    Generalizes across architectures (MLP, CNN, ResNet) via the ``(fan_in, fan_out)``
    weight convention. ``Linear`` / ``Conv2D`` get full signal/gain/budget
    calibration; ``BatchNorm2D`` gets the degenerate per-channel calibration.

    Args:
        net:        A ``Sequential`` network (already on its device).
        sigma2_obs: Seed observation-noise variance (adapted online afterward).
        metric:     ``"analytic"`` — the closed-form ReLU fixed point (``Gᶜ = v·I``),
                    no data needed, all architectures. ``"data"`` — data-driven
                    whitening: the per-layer metric ``Gᶜ`` is the *measured* Gram of
                    real activations, so signal = 1 holds on the true input
                    distribution rather than the ReLU-fixed-point assumption.
                    ``"data"`` requires ``data_batch`` and currently supports dense
                    (``Linear``) layers only (see ``_calibrate_data``).
        input_var:  Assumed variance of the (normalized) network input (analytic only).
        seed:       RNG seed for the orthonormal seeds (reproducibility).
        data_batch: Calibration inputs ``(N, …)`` on the network device, used to
                    measure the empirical metric when ``metric="data"``. Ignored by
                    the analytic metric. A few hundred–thousand samples suffice.

    Returns:
        Per-layer metadata list (also attached as ``layer._calib``), consumed by
        :func:`online_recalibrate`.

    Raises:
        ValueError:          unknown ``metric``, or ``metric="data"`` without ``data_batch``.
        NotImplementedError: ``metric="data"`` on a network with non-dense learnable
                             layers (Conv2D / BatchNorm2D / ResBlock).
    """
    if metric == "data":
        if data_batch is None:
            raise ValueError("metric='data' requires a calibration `data_batch`.")
        return _calibrate_data(net, sigma2_obs=sigma2_obs, data_batch=data_batch, seed=seed)
    if metric != "analytic":
        raise ValueError(f"Unknown metric: {metric!r} (expected 'analytic' or 'data').")

    Vz = 1.0 + sigma2_obs
    metas: list[dict] = []
    first_matmul = True

    for layer in _iter_learnable(net.layers):
        device = layer.mw.device
        gen = _make_generator(seed, device, len(metas))

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
                has_bias=getattr(layer, "has_bias", True), unwhiten=math.sqrt(v), generator=gen,
            )

        layer._calib = meta
        metas.append(meta)

    return metas


# ----------------------------------------------------------------------
#  Data-driven (empirical) metric — measured Gram whitening (LSUV/ZCA style)
# ----------------------------------------------------------------------


def _propagate_det(layer, A: Tensor) -> Tensor:
    """Deterministic mean-propagation of calibration samples through one layer.

    Pushes *real* samples (no variance) through ``layer`` so the next dense layer
    sees the true activation distribution. ReLU is applied exactly; other stateless
    layers reuse their moment forward with zero input variance (its mean output is
    the deterministic transform).
    """
    from .layers.relu import ReLU

    if isinstance(layer, ReLU):
        return torch.relu(A)
    ma, _ = layer.forward(A, torch.zeros_like(A))
    return ma


def _calibrate_data(net, *, sigma2_obs: float, data_batch: Tensor, seed: int | None) -> list[dict]:
    """Calibrate dense layers from the *measured* Gram of propagated real samples.

    Walks ``net.layers`` in order, threading a deterministic sample matrix ``A``.
    At each ``Linear`` layer it measures the empirical metric of the incoming
    activations — ``Ea = E[a]``, ``g_diag = E[a²]`` (diag of the Gram), and the
    centred Gram ``Gᶜ = Cov(a)`` with whitener ``(Gᶜ)^{-1/2}`` — then applies the
    same operator T (signal = 1, balanced gain, budget = σ_v²) and propagates the
    calibrated mean forward. Mirrors the ``:empirical`` path of ``proof_unified.jl``.

    Conv2D / BatchNorm2D / ResBlock are not supported on the data path (the Gram
    would be over im2col patches / per-channel affines); use ``metric="analytic"``
    for convolutional networks.
    """
    from .layers.linear import Linear

    metas: list[dict] = []
    learnable = _iter_learnable(net.layers)
    if any(not isinstance(layer, Linear) for layer in learnable):
        raise NotImplementedError(
            "metric='data' currently supports dense (Linear) networks only; the "
            "network has Conv2D/BatchNorm2D/ResBlock layers. Use metric='analytic'."
        )

    device = learnable[0].mw.device
    A = data_batch.to(device)

    for layer in net.layers:
        if isinstance(layer, Linear):
            A_flat = A.reshape(A.shape[0], -1)
            N, fan_in = A_flat.shape
            if N < 2:
                raise ValueError("metric='data' needs at least 2 samples to estimate a Gram.")

            Ea = A_flat.mean(dim=0)                          # E[a]      (fan_in,)
            g_diag = (A_flat * A_flat).mean(dim=0)           # E[a²]     (fan_in,)  diag(G)
            Ac = A_flat - Ea                                 # centred activations
            Gc = (Ac.transpose(0, 1) @ Ac) / N               # Cov(a) = Gᶜ  (fan_in × fan_in)
            unwhiten, whiten = sqrt_invsqrt_psd(Gc, eps=_REG_EPS)   # (Gᶜ)^{±1/2}

            gen = _make_generator(seed, device, len(metas))
            meta = calibrate_dense_block(
                layer.mw, layer.Sw, getattr(layer, "mb", None), getattr(layer, "Sb", None),
                Ea=Ea, g_diag=g_diag, whiten=whiten, v_analytic=1.0, sigma2=sigma2_obs,
                has_bias=getattr(layer, "has_bias", True), unwhiten=unwhiten, generator=gen,
            )
            layer._calib = meta
            metas.append(meta)

            # Propagate the calibrated mean to the next layer (ReLU applied by it).
            A = A_flat @ layer.mw + layer.mb
        else:
            A = _propagate_det(layer, A)

    return metas


# ======================================================================
#  Online operator T (per-step): re-inflation + mean projection
# ======================================================================


def surprise_lambda(chi2: float, lam_max: float, tau: float) -> float:
    """Map batch surprise χ² to a forgetting rate λ ∈ [0, lam_max]."""
    excess = max(chi2 - 1.0, 0.0)
    return lam_max * (1.0 - math.exp(-excess / max(tau, 1e-8)))


def batch_chi2(y: Tensor, y_pred_mu: Tensor, y_pred_var: Tensor, sigma2: float) -> float:
    """Mean normalized innovation ``E[(y-μ)²/(Var+σ_v²)]`` on a batch.

    With a TAGI-V / AGVI head (output width = 2·target), the noise is the *predicted
    aleatoric* variance (odd columns), so the denominator is
    ``Var_z[even] + E[V²][odd]`` — the same total variance AGVI uses for the mean
    update — not the (here meaningless) scalar ``σ_v²``. Otherwise the scalar
    ``σ_v²`` is used via the Triton innovation map.
    """
    if y_pred_mu.shape[-1] == 2 * y.shape[-1]:         # TAGI-V head: [m0,v0,m1,v1,...]
        mu = y_pred_mu[..., 0::2]
        var_ep = y_pred_var[..., 0::2]                 # epistemic Var[z] (mean head)
        var_al = y_pred_mu[..., 1::2]                  # aleatoric E[V²] (AGVI noise head)
        denom = (var_ep + var_al).clamp_min(1e-12)
        return float((((y - mu) ** 2) / denom).mean().item())
    return float(normalized_innovation(y, y_pred_mu, y_pred_var, sigma2).mean().item())


def _reinflate_layer(layer, lam: float, sigma_v2: float) -> None:
    """(S) Re-inflate a layer's posterior variance toward its calibrated target.

    The target gain ``c = σ_v² / budget_denom`` is recomputed from the *current*
    ``sigma_v2`` (not the frozen init seed), so condition (III) — Σ epistemic = σ_v²
    ⇒ output Kalman gain J = ½ — stays satisfied as the online σ_v² adapts.
    """
    if lam <= 0:
        return
    meta = layer._calib
    c = sigma_v2 / meta["budget_denom"]
    if meta["kind"] == "norm":
        reinflate_const_(layer.Sw, c, lam)
        reinflate_const_(layer.Sb, c, lam)
        return
    reinflate_weight_(layer.Sw, meta["sqrt_g"], c, lam)
    if getattr(layer, "has_bias", True) and getattr(layer, "Sb", None) is not None:
        reinflate_const_(layer.Sb, c, lam)


def _project_layer(layer, beta: float) -> None:
    """(μ) Muon: pull mw a fraction β toward the signal=1 manifold, re-centre bias.

    Orthogonalization is the EXACT polar factor (SVD) — see ``polar_orth_columns``.
    """
    if beta <= 0:
        return
    meta = layer._calib
    if meta["kind"] == "norm":
        return
    fan_in, fan_out = meta["fan_in"], meta["fan_out"]
    if fan_in < fan_out:                               # projection ill-posed; reinflation only
        return
    whiten = meta["whiten"]
    unwhiten = meta["unwhiten"]                        # (Gᶜ)^{1/2}, stored at calibration
    mw = layer.mw
    if isinstance(whiten, float):
        B = unwhiten * mw                              # B Bᵀ ∝ mwᵀ Gᶜ mw  (signal block)
        proj = whiten * polar_orth_columns(B)
    else:
        B = unwhiten @ mw
        proj = whiten @ polar_orth_columns(B)
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
        project:     Apply the Muon mean projection (exact SVD polar factor).
        beta:        Mean-projection relaxation rate.
        every:       Batches between projections.
        sigma_v_mode: ``"heteroscedastic"`` — the TAGI-V / AGVI latent-variance head
                      learns σ_v² (Bayesian; σ_v² state here is ignored) — or
                      ``"homoscedastic"`` — a method-of-moments running σ_v²
                      estimate (non-AGVI fallback for nets without a V2 head).
        sigma_v2:    Running σ_v² estimate (mutable).
        sigma_v2_rho: EMA rate for the σ_v² estimate.
        sigma_v2_floor: Lower bound on σ_v².
        history_max: Max number of recent λ values to retain in ``lambda_hist``
                     (``None`` = unbounded). Bounded by default to cap memory on
                     long runs; per-step λ is always available via ``last_lambda``.
    """

    _MODES = ("off", "const", "surprise")
    _SIGMA_V_MODES = ("homoscedastic", "heteroscedastic")

    mode: str = "surprise"
    lam: float = 0.05
    tau: float = 1.0
    project: bool = True
    beta: float = 0.10
    every: int = 25
    sigma_v_mode: str = "homoscedastic"
    sigma_v2: float = 0.01
    sigma_v2_rho: float = 0.01
    sigma_v2_floor: float = 1e-4
    history_max: int | None = 10_000
    step_count: int = 0
    last_lambda: float = 0.0
    lambda_hist: deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        # Fail fast on misconfigured string enums instead of silently no-op'ing.
        if self.mode not in self._MODES:
            raise ValueError(f"mode must be one of {self._MODES}, got {self.mode!r}")
        if self.sigma_v_mode not in self._SIGMA_V_MODES:
            raise ValueError(
                f"sigma_v_mode must be one of {self._SIGMA_V_MODES}, got {self.sigma_v_mode!r}"
            )
        # Bound the λ history so long runs don't accumulate unboundedly.
        self.lambda_hist = deque(self.lambda_hist, maxlen=self.history_max)

    @property
    def sigma_v(self) -> float:
        """Current observation-noise std used by the innovation."""
        return math.sqrt(max(self.sigma_v2, self.sigma_v2_floor))


def update_sigma_v2(cfg: OnlineCalibration, y: Tensor, y_pred_mu: Tensor, y_pred_var: Tensor) -> None:
    """Method-of-moments σ_v² fallback: σ̂² = E[(y-μ)²] − E[Var_z] (EMA).

    A lightweight non-AGVI estimator for nets without a TAGI-V / AGVI variance head.
    Skipped in heteroscedastic mode, where the AGVI head infers the noise instead.
    """
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
    sigma_v2 = max(cfg.sigma_v2, cfg.sigma_v2_floor)   # live budget (tracks online σ_v²)
    do_proj = cfg.project and (cfg.step_count % max(cfg.every, 1) == 0)
    for layer in _iter_learnable(net.layers):
        if getattr(layer, "_calib", None) is None:
            continue
        _reinflate_layer(layer, lam, sigma_v2)         # (S)
        if do_proj:
            _project_layer(layer, cfg.beta)            # (μ) Muon — exact SVD polar factor
    cfg.last_lambda = lam
    cfg.lambda_hist.append(lam)
