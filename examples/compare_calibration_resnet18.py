"""
Old vs. new methodology on CIFAR-10 ResNet-18 — triton-tagi.

Same deep ResNet-18 architecture for both arms; the ONLY difference is the
calibration. Noise is learned by **TAGI-V / AGVI** in both arms (a 2·K output head
+ EvenSoftplus), so the comparison isolates the calibration effect, not the
noise-learning method:

    baseline (old)                       calibrated (new)
    ---------------------------------    -----------------------------------------
    He init + gain_w/gain_b              operator T init: signal=1, K_b/K_w=1, J=½
    no re-inflation → variance collapse  surprise-driven re-inflation (process Q)
    μ_W drifts → outputs correlate       Muon exact-SVD polar projection
    TAGI-V noise head (AGVI)             TAGI-V noise head (AGVI)   ← identical

Reuses the ResNet-18 backbone and CIFAR-10 loader from
``cifar10_resnet18_calibrated.py`` and swaps the output head to TAGI-V. See
``docs/calibration.md`` for the math.

Usage:
    python examples/compare_calibration_resnet18.py
    python examples/compare_calibration_resnet18.py --n_epochs 30 --no_augment
"""

from __future__ import annotations

import argparse
import time

import torch
from cifar10_resnet18_calibrated import build_resnet18, gpu_augment, load_cifar10

from triton_tagi import (
    EvenSoftplus,
    Linear,
    OnlineCalibration,
    Sequential,
    calibrate,
)
from triton_tagi.base import LearnableLayer

N_CLASSES = 10


def with_tagiv_head(net: Sequential, device: torch.device) -> Sequential:
    """Swap a net's classification head for a TAGI-V head (keep the backbone).

    Replaces the trailing ``Linear(512, 10) → Remax`` with
    ``Linear(512, 2·10) → EvenSoftplus(10)``: interleaved [mean, noise-var] per
    class, so the observation noise is learned by TAGI-V / AGVI.
    """
    in_features = net.layers[-2].in_features            # the final Linear's fan-in (512)
    net.layers = net.layers[:-2] + [
        Linear(in_features, 2 * N_CLASSES, device=device),
        EvenSoftplus(half_width=N_CLASSES),
    ]
    return net


def evaluate(net: Sequential, x_test, y_labels, batch_size: int = 256) -> float:
    """Test accuracy from the TAGI-V mean head (even columns)."""
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            mu, _ = net.forward(x_test[i : i + batch_size])
            pred = mu[:, 0::2].argmax(dim=1)            # even cols = class means
            correct += (pred == y_labels[i : i + batch_size]).sum().item()
    net.train()
    return correct / len(x_test)


def deep_layer_weight_var(net: Sequential) -> float:
    """Mean weight variance σ²_W of a middle learnable layer (collapse probe)."""
    learn = [layer for layer in _flatten_learnable(net) if getattr(layer, "Sw", None) is not None]
    mid = learn[len(learn) // 2]
    return float(mid.Sw.mean().item())


def _flatten_learnable(net: Sequential) -> list:
    out: list = []
    for layer in net.layers:
        sub = getattr(layer, "_learnable", None)
        if sub is not None:
            out.extend(sub)
        elif isinstance(layer, LearnableLayer):
            out.append(layer)
    return out


def train_one(net, online, x_train, y_oh, x_test, y_labels,
              n_epochs, batch_size, sigma_v, augment, device):
    """Train one arm. Returns (best_acc, per-epoch acc, per-epoch deep-layer σ²_W)."""
    accs: list[float] = []
    wvar: list[float] = []
    best = 0.0
    for _epoch in range(1, n_epochs + 1):
        perm = torch.randperm(x_train.size(0), device=device)
        x_s, y_s = x_train[perm], y_oh[perm]
        for i in range(0, len(x_s), batch_size):
            xb = x_s[i : i + batch_size]
            if augment:
                xb = gpu_augment(xb)
            net.step(xb, y_s[i : i + batch_size], sigma_v, online=online)
        if device.type == "cuda":
            torch.cuda.synchronize()
        acc = evaluate(net, x_test, y_labels)
        accs.append(acc)
        wvar.append(deep_layer_weight_var(net))
        best = max(best, acc)
    return best, accs, wvar


def main(
    n_epochs: int = 30,
    batch_size: int = 128,
    sigma_v: float = 0.05,
    sigma2_obs: float = 0.01,
    augment: bool = True,
    data_dir: str = "data",
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    torch.manual_seed(seed)
    dev = torch.device(device)

    print("=" * 70)
    print("  CIFAR-10 ResNet-18 — old vs new (both with TAGI-V / AGVI noise head)")
    print("=" * 70)
    if dev.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

    print(f"\n  Loading CIFAR-10 from '{data_dir}'...", flush=True)
    x_train, y_train_oh, x_test, y_test_labels = load_cifar10(data_dir, dev)

    # ── Both arms: identical ResNet-18 backbone + TAGI-V head, same seed ──
    torch.manual_seed(seed)
    net_base = with_tagiv_head(build_resnet18(dev), dev)
    torch.manual_seed(seed)
    net_cal = with_tagiv_head(build_resnet18(dev), dev)

    # New methodology: operator T init + online (heteroscedastic → TAGI-V owns σ_v).
    calibrate(net_cal, sigma2_obs=sigma2_obs, metric="analytic")
    online = OnlineCalibration(mode="surprise", project=True, sigma_v_mode="heteroscedastic")

    print(f"\n  Params: {net_cal.num_parameters():,}  |  epochs={n_epochs} batch={batch_size}")
    print("  baseline = He init, no calibration   |   calibrated = operator T + online\n")

    t0 = time.perf_counter()
    base_best, base_acc, base_wv = train_one(
        net_base, None, x_train, y_train_oh, x_test, y_test_labels,
        n_epochs, batch_size, sigma_v, augment, dev)
    cal_best, cal_acc, cal_wv = train_one(
        net_cal, online, x_train, y_train_oh, x_test, y_test_labels,
        n_epochs, batch_size, sigma_v, augment, dev)
    wall = time.perf_counter() - t0

    wv0_b, wv0_c = base_wv[0], cal_wv[0]
    print(f"  {'Epoch':>5} │ {'base acc':>9}  {'calib acc':>9} │"
          f" {'base σ²_W↓':>11}  {'calib σ²_W↓':>11}")
    print("  ──────┼" + "─" * 22 + "┼" + "─" * 26)
    for ep in range(n_epochs):
        print(f"  {ep + 1:5d} │ {base_acc[ep] * 100:8.2f}%  {cal_acc[ep] * 100:8.2f}% │"
              f" {base_wv[ep] / wv0_b:11.3f}  {cal_wv[ep] / wv0_c:11.3f}")

    print("\n" + "=" * 70)
    print(f"  best test acc   baseline = {base_best * 100:6.2f}%   "
          f"calibrated = {cal_best * 100:6.2f}%")
    print(f"  deep-layer σ²_W  baseline ×{base_wv[-1] / wv0_b:.3f}   "
          f"calibrated ×{cal_wv[-1] / wv0_c:.3f}   (1.0 = no collapse)")
    print(f"  total wall = {wall:.1f}s")
    print("=" * 70)
    return {"base_best": base_best, "cal_best": cal_best,
            "base_acc": base_acc, "cal_acc": cal_acc}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CIFAR-10 ResNet-18 baseline vs calibrated (TAGI-V)")
    p.add_argument("--n_epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--sigma_v", type=float, default=0.05, help="baseline seed (ignored by TAGI-V)")
    p.add_argument("--sigma2_obs", type=float, default=0.01, help="calibration σ_v² seed")
    p.add_argument("--no_augment", dest="augment", action="store_false")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.set_defaults(augment=True)
    args = p.parse_args()
    main(**vars(args))
