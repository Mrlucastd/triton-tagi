"""
CIFAR-10 — ResNet-18 — Unified parameter-free online calibration (triton-tagi).

This is the calibrated counterpart of ``cifar10_resnet18.py``. It removes every
init/training hyperparameter the baseline relies on:

    baseline                         calibrated (this file)
    ----------------------------     -----------------------------------------
    init_method="He" / gain_w/gain_b unified operator T  (calibrate(net))
    sigma_v = 0.05 (+ decay)         learned online (homoscedastic running σ_v²)
    no re-inflation (collapse)       online operator T: surprise-driven λ_t
                                     re-inflation + Muon mean projection

The network is built with default layer init, then ``calibrate(net)`` overwrites
every parameter so that, at every layer: signal = 1, the epistemic budget equals
σ_v² (output Kalman gain J = ½), and per-parameter gains are balanced
(K_b/K_w = 1). During training the online operator keeps those invariants alive
— the only mechanism that prevents TAGI's monotone variance collapse — with the
forgetting rate set automatically from the batch surprise χ².

Usage:
    python examples/cifar10_resnet18_calibrated.py
    python examples/cifar10_resnet18_calibrated.py --n_epochs 30 --no_augment
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from triton_tagi import (
    AvgPool2D,
    BatchNorm2D,
    Conv2D,
    Flatten,
    Linear,
    OnlineCalibration,
    ReLU,
    Remax,
    ResBlock,
    Sequential,
    calibrate,
)
from triton_tagi.checkpoint import RunDir

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


def load_cifar10(data_dir: str, device: torch.device):
    """Load CIFAR-10 as normalized (N,3,32,32) tensors on ``device``."""
    norm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)]
    )
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=norm)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=norm)

    x_train = torch.stack([img for img, _ in train_ds]).to(device)
    y_train = torch.tensor([lbl for _, lbl in train_ds], device=device)
    x_test = torch.stack([img for img, _ in test_ds]).to(device)
    y_test = torch.tensor([lbl for _, lbl in test_ds], device=device)

    y_train_oh = torch.zeros(len(y_train), 10, device=device)
    y_train_oh.scatter_(1, y_train.unsqueeze(1), 1.0)
    return x_train, y_train_oh, x_test, y_test


def gpu_augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random horizontal flip + random crop on-device."""
    B, C, H, W = x.shape
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    top = torch.randint(0, 2 * pad, (B,), device=x.device)
    left = torch.randint(0, 2 * pad, (B,), device=x.device)
    rows = top.unsqueeze(1) + torch.arange(H, device=x.device).unsqueeze(0)
    cols = left.unsqueeze(1) + torch.arange(W, device=x.device).unsqueeze(0)
    return x_pad[
        torch.arange(B, device=x.device)[:, None, None, None],
        torch.arange(C, device=x.device)[None, :, None, None],
        rows[:, None, :, None].expand(B, C, H, W),
        cols[:, None, None, :].expand(B, C, H, W),
    ]


def evaluate(net: Sequential, x_test, y_labels, batch_size: int = 256) -> float:
    """Return test accuracy."""
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            mu, _ = net.forward(x_test[i : i + batch_size])
            correct += (mu.argmax(dim=1) == y_labels[i : i + batch_size]).sum().item()
    net.train()
    return correct / len(x_test)


def build_resnet18(device: torch.device) -> Sequential:
    """CIFAR-10 ResNet-18 (default init; calibrate() overwrites all parameters)."""
    kw = {"device": device}
    return Sequential(
        [
            Conv2D(3, 64, 3, stride=1, padding=1, **kw),
            ReLU(),
            BatchNorm2D(64, **kw),
            ResBlock(64, 64, stride=1, **kw),
            ResBlock(64, 64, stride=1, **kw),
            ResBlock(64, 128, stride=2, **kw),
            ResBlock(128, 128, stride=1, **kw),
            ResBlock(128, 256, stride=2, **kw),
            ResBlock(256, 256, stride=1, **kw),
            ResBlock(256, 512, stride=2, **kw),
            ResBlock(512, 512, stride=1, **kw),
            AvgPool2D(4),
            Flatten(),
            Linear(512, 10, **kw),
            Remax(),
        ],
        device=device,
    )


def train(net, online, x_train, y_train_oh, x_test, y_test_labels,
          n_epochs, batch_size, augment, device, run, config) -> float:
    """Parameter-free online training loop. Returns best test accuracy."""
    print(f"\n  {'Epoch':>5}  {'Test Acc':>9}  {'σ_v':>8}  {'mean λ':>8}  {'Time':>7}")
    print("  " + "─" * 50)
    best_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(x_train.size(0), device=device)
        x_s, y_s = x_train[perm], y_train_oh[perm]
        ep_lam_sum, ep_lam_n = 0.0, 0

        for i in range(0, len(x_s), batch_size):
            xb = x_s[i : i + batch_size]
            if augment:
                xb = gpu_augment(xb)
            # sigma_v arg is ignored in homoscedastic mode (online owns σ_v).
            net.step(xb, y_s[i : i + batch_size], 0.0, online=online)
            ep_lam_sum += online.last_lambda             # running mean λ for this epoch
            ep_lam_n += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        acc = evaluate(net, x_test, y_test_labels)
        best_acc = max(best_acc, acc)
        mean_lam = ep_lam_sum / max(ep_lam_n, 1)
        print(f"  {epoch:5d}  {acc*100:8.2f}%  {online.sigma_v:8.4f}  {mean_lam:8.4f}  {wall:6.2f}s")
        run.append_metrics(epoch, test_acc=acc, sigma_v=online.sigma_v, mean_lambda=mean_lam, wall_s=wall)

        if epoch % config.get("checkpoint_interval", 10) == 0 or epoch == n_epochs:
            run.save_checkpoint(net, epoch, config)

    print("  " + "─" * 50)
    print(f"  Best test accuracy: {best_acc*100:.2f}%")
    return best_acc


def main(
    n_epochs: int = 100,
    batch_size: int = 128,
    sigma2_obs: float = 0.01,
    augment: bool = True,
    data_dir: str = "data",
    checkpoint_interval: int = 10,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    """CIFAR-10 ResNet-18 with unified parameter-free online calibration.

    Args:
        sigma2_obs: Seed observation-noise variance (learned online thereafter).
        augment:    Random flip + crop augmentation each batch.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)

    print("=" * 60)
    print("  CIFAR-10 — ResNet-18 — unified parameter-free calibration")
    print("=" * 60)
    if dev.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

    print(f"\n  Loading CIFAR-10 from '{data_dir}'...", flush=True)
    x_train, y_train_oh, x_test, y_test_labels = load_cifar10(data_dir, dev)
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")

    config: dict = {
        "dataset": "cifar10",
        "arch": "resnet18",
        "optimizer": "tagi-calibrated",
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "sigma2_obs_seed": sigma2_obs,
        "augment": augment,
        "checkpoint_interval": checkpoint_interval,
        "seed": seed,
        "device": device,
    }
    run = RunDir("cifar10", "resnet18_calibrated", "tagi")
    run.save_config(config)
    print(f"  Run directory: {run.path}")

    net = build_resnet18(dev)

    # ── Unified operator T (init): signal=1, balanced gains, budget=σ_v² ──
    calibrate(net, sigma2_obs=sigma2_obs, metric="analytic")
    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")

    # ── Online operator T: surprise-driven λ + Muon projection + online σ_v² ──
    online = OnlineCalibration(
        mode="surprise",
        sigma_v_mode="homoscedastic",
        sigma_v2=sigma2_obs,
        project=True,
    )
    print(f"  Online: mode={online.mode}, project={online.project}, "
          f"sigma_v_mode={online.sigma_v_mode}  (parameter-free)")

    best_acc = train(
        net, online, x_train, y_train_oh, x_test, y_test_labels,
        n_epochs, batch_size, augment, dev, run, config,
    )
    print(f"\n  Results in: {run.path}")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 ResNet-18 — parameter-free calibration")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sigma2_obs", type=float, default=0.01, help="σ_v² seed (learned online)")
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.set_defaults(augment=True)
    args = parser.parse_args()
    main(**vars(args))
