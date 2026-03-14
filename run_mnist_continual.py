"""
MNIST continual-learning validation with TAGISNNNetworkVec (vectorized).

Training is class-incremental and strictly sequential: 0, then 1, ..., then 9.
"""

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import csv
import torch
from torchvision import datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util

def _load_vec_class():
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "src", "snn_tagi_vec.py")
    spec = importlib.util.spec_from_file_location("snn_tagi_vec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TAGISNNNetworkVec

TAGISNNNetworkVec = _load_vec_class()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential MNIST continual learning with TAGISNN")
    parser.add_argument("--data-dir", type=str, default="data", help="MNIST root directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--epochs-per-class", type=int, default=1, help="Epochs for each class")
    parser.add_argument("--dt-train", type=float, default=0.2, help="Poisson dt during training")
    parser.add_argument("--dt-eval", type=float, default=0.2, help="Poisson dt during eval")
    parser.add_argument("--density", type=float, default=0.01, help="Initial graph density")
    parser.add_argument("--sleep-every", type=int, default=5000, help="Steps between sleep phases")
    parser.add_argument("--n-steps-train", type=int, default=1, help="Timesteps per sample (training)")
    parser.add_argument("--n-steps-eval", type=int, default=3, help="Timesteps per sample (eval)")
    parser.add_argument("--max-train-per-class", type=int, default=0, help="Train cap per class (0=no cap)")
    parser.add_argument("--max-test-per-class", type=int, default=0, help="Test cap per class (0=no cap)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device",
    )
    return parser.parse_args()


def load_mnist(data_dir: str, device: torch.device):
    train_ds = datasets.MNIST(data_dir, train=True, download=True)
    test_ds = datasets.MNIST(data_dir, train=False, download=True)

    x_train = train_ds.data.float().view(-1, 784) / 255.0
    x_test = test_ds.data.float().view(-1, 784) / 255.0

    x_train = x_train.to(device)
    x_test = x_test.to(device)
    y_train = train_ds.targets.to(device)
    y_test = test_ds.targets.to(device)

    return x_train, y_train, x_test, y_test


def build_net(
    device: torch.device,
    n_input: int,
    n_output: int,
    density: float,
    n_hidden_pool: int = 200,
) -> TAGISNNNetworkVec:
    """
    Build the network with:
      - n_input  input  neurons  [0 .. n_input)
      - n_hidden dormant hidden neurons  [n_input .. n_input+n_hidden_pool)
      - n_output output neurons  [n_input+n_hidden_pool .. n_input+n_hidden_pool+n_output)

    Hidden neurons start DORMANT and are recruited by neurogenesis when
    the model encounters observations it cannot explain.
    """
    n_hidden = n_hidden_pool
    return TAGISNNNetworkVec(
        n_neurons=n_input + n_hidden + n_output,
        input_ids=list(range(n_input)),
        output_ids=list(range(n_input + n_hidden, n_input + n_hidden + n_output)),
        device=device,
        prior_var=0.1,
        density=density,
        ensure_io=True,
        # Neurogenesis: activate up to 5 hidden neurons per high-error step
        neurogenesis_surprise_threshold=1.5,
        neurogenesis_n_per_event_max=5,
        neurogenesis_n_inputs_per_neuron=50,
        neurogenesis_ema_alpha=0.05,
        neurogenesis_cooldown=300,
    )


def predict_class(
    net: TAGISNNNetworkVec,
    rates_784: torch.Tensor,
    input_ids: List[int],
    output_ids: List[int],
    dt: float,
    n_steps: int = 3,
) -> int:
    """Run n_steps forward passes and pick the class with highest mean voltage."""
    out_idx = torch.tensor(output_ids, device=rates_784.device, dtype=torch.long)
    all_rates = torch.zeros(net.N, device=rates_784.device)
    all_rates[input_ids] = rates_784

    # Reset voltage / spike-history so this sample is evaluated independently
    net.reset_dynamics()
    mu_accum = torch.zeros(len(output_ids), device=rates_784.device)
    for _ in range(n_steps):
        S = net.poisson_encoding(all_rates, dt=dt)
        net.forward_step(S, targets=None)
        mu_accum += net.mu_V[out_idx]

    return int(mu_accum.argmax().item())


def evaluate_per_class(
    net: TAGISNNNetworkVec,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    classes: List[int],
    input_ids: List[int],
    output_ids: List[int],
    dt_eval: float,
    n_steps: int = 3,
    max_per_class: int = 0,
) -> Dict[int, float]:
    per_class_acc: Dict[int, float] = {}

    for c in classes:
        idx = torch.where(y_test == c)[0]
        if idx.numel() == 0:
            per_class_acc[c] = float("nan")
            continue

        # Shuffle test indices for regular randomized evaluation
        perm = torch.randperm(idx.numel(), device=idx.device)
        idx = idx[perm]

        if max_per_class > 0 and idx.numel() > max_per_class:
            idx = idx[:max_per_class]

        correct = 0
        total = int(idx.numel())
        for i in idx.tolist():
            pred = predict_class(net, x_test[i], input_ids, output_ids, dt_eval, n_steps)
            correct += int(pred == int(y_test[i].item()))

        per_class_acc[c] = correct / max(total, 1)

    return per_class_acc


def main():
    args = parse_args()
    device = torch.device(args.device)

    n_input  = 784
    n_output = 10

    print("=" * 72)
    print("MNIST Continual Learning with TAGISNNNetworkVec (vectorized)")
    print("Order: 0 -> 1 -> ... -> 9")
    print("=" * 72)
    print(f"Device: {device}")

    x_train, y_train, x_test, y_test = load_mnist(args.data_dir, device)
    net = build_net(device, n_input, n_output, args.density, n_hidden_pool=500)

    # Derive ids from the network so they always match the hidden-pool layout
    input_ids  = net.input_ids
    output_ids = net.output_ids
    n_hidden   = len(net.hidden_ids)

    print(f"Train: {x_train.shape[0]:,} | Test: {x_test.shape[0]:,}")
    print(
        f"Neurons: {net.n_neurons} "
        f"(inputs={n_input}, hidden_pool={n_hidden} dormant, outputs={n_output})"
    )
    print(
        f"epochs/class={args.epochs_per_class}, batch={args.batch_size}, "
        f"dt_train={args.dt_train}, dt_eval={args.dt_eval}, density={args.density}"
    )
    print(f"n_steps_train={args.n_steps_train}, n_steps_eval={args.n_steps_eval}")

    # Prepare plots/metrics directory
    plots_dir = os.path.join("plots", "continual")
    os.makedirs(plots_dir, exist_ok=True)
    metrics_path = os.path.join(plots_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "task_idx", "class", "train_time_s", "n_connections", "delta_connections",
                "n_dead", "n_mature", "n_active", "n_dormant", "n_silent",
                "n_newly_spiking", "neurons_grown_total", "surprise_ema",
                "mean_theta", "mean_obs_var", "mean_seen_acc"
            ])

    class_order = list(range(10))
    acc_matrix: List[Dict[int, float]] = []
    learn_step_of_class = {c: i for i, c in enumerate(class_order)}

    t_global = time.perf_counter()

    out_idx = torch.tensor(output_ids, device=device, dtype=torch.long)

    for task_idx, cls in enumerate(class_order):
        print("\n" + "-" * 72)
        print(f"Task {task_idx + 1}/10 - train class {cls}")

        idx = torch.where(y_train == cls)[0]
        if args.max_train_per_class > 0 and idx.numel() > args.max_train_per_class:
            perm = torch.randperm(idx.numel(), device=device)
            idx = idx[perm[:args.max_train_per_class]]

        x_task = x_train[idx]
        y_task = y_train[idx]

        # Snapshot network state before this task for analytics
        n_conn_before = int(net.mask.sum().item())
        spike_counts_before = net.spike_counts.clone()

        t_task = time.perf_counter()
        for ep in range(args.epochs_per_class):
            perm = torch.randperm(x_task.shape[0], device=device)
            x_ep = x_task[perm]
            y_ep = y_task[perm]

            for b in range(0, x_ep.shape[0], args.batch_size):
                xb = x_ep[b:b + args.batch_size]
                yb = y_ep[b:b + args.batch_size]

                for k in range(xb.shape[0]):
                    all_rates = torch.zeros(net.N, device=device)
                    all_rates[input_ids] = xb[k]

                    label = int(yb[k].item())
                    targets = {
                        output_ids[c]: (1.0 if c == label else 0.0)
                        for c in range(n_output)
                    }

                    for t in range(args.n_steps_train):
                        S = net.poisson_encoding(all_rates, dt=args.dt_train)
                        net.forward_step(S, targets=targets)

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_task = time.perf_counter() - t_task
        stats = net.get_network_stats()
        # Connection changes and neuron activation stats
        n_conn_after = int(net.mask.sum().item())
        delta_conn = n_conn_after - n_conn_before
        newly_activated = int(((net.spike_counts > 0) & (spike_counts_before == 0)).sum().item())
        print(
            f"  train time: {dt_task:.2f}s | θ_out={stats['mean_theta']:.3f} | "
            f"obs_var={stats['mean_obs_var']:.6f} | Δconn={delta_conn} | "
            f"newly_spiking={newly_activated} | "
            f"neurons_grown={stats['neurogenesis_total']} | "
            f"dormant_left={stats['n_dormant']} | "
            f"err_ema={stats['surprise_ema']:.3f}"
        )

        # ── Evaluate on seen classes ────────────────────────────────
        seen = class_order[:task_idx + 1]
        eval_max = args.max_test_per_class if args.max_test_per_class > 0 else 200
        acc_seen = evaluate_per_class(
            net, x_test, y_test, seen, input_ids, output_ids,
            args.dt_eval, n_steps=args.n_steps_eval, max_per_class=eval_max,
        )
        acc_matrix.append(acc_seen)

        mean_seen = float(np.mean([acc_seen[c] for c in seen]))
        summary = " | ".join([f"{c}:{acc_seen[c] * 100:5.1f}%" for c in seen])
        print(f"  Seen-class acc: {summary}")
        print(f"  Mean seen acc: {mean_seen * 100:5.2f}%")

        # Append metrics to CSV (after evaluation)
        try:
            with open(metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    task_idx, cls, f"{dt_task:.4f}", n_conn_after, delta_conn,
                    stats.get("n_dead", 0), stats.get("n_mature", 0),
                    stats.get("n_active", 0), stats.get("n_dormant", 0), stats.get("n_silent", 0),
                    newly_activated, stats.get("neurogenesis_total", 0),
                    f"{stats.get('surprise_ema', 0.0):.6f}",
                    f"{stats['mean_theta']:.6f}", f"{stats['mean_obs_var']:.8f}", f"{mean_seen:.6f}"
                ])
        except Exception:
            pass

    # ── Final report ────────────────────────────────────────────────
    final_acc = acc_matrix[-1]
    forgetting = {}
    for c in class_order:
        start = learn_step_of_class[c]
        best_after_learn = max(step.get(c, 0.0) for step in acc_matrix[start:])
        forgetting[c] = best_after_learn - final_acc.get(c, 0.0)

    avg_forget_all = float(np.mean(list(forgetting.values())))
    avg_forget_0_8 = float(np.mean([forgetting[c] for c in class_order[:-1]]))

    total_time = time.perf_counter() - t_global

    print("\n" + "=" * 72)
    print("Final continual-learning report")
    print("=" * 72)
    print("Final per-class accuracy:")
    print(" | ".join([f"{c}:{final_acc.get(c, 0.0) * 100:5.1f}%" for c in class_order]))
    print("Forgetting per class (best-after-learn - final):")
    print(" | ".join([f"{c}:{forgetting[c] * 100:5.1f}%" for c in class_order]))
    print(f"Average forgetting (all classes): {avg_forget_all * 100:5.2f}%")
    print(f"Average forgetting (classes 0-8): {avg_forget_0_8 * 100:5.2f}%")
    print(f"Total run time: {total_time:.2f}s")
    
    # ── Plotting ────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        
        # Plot each class accuracy over tasks
        for c in class_order:
            start_task = learn_step_of_class[c]
            accs = [step.get(c, 0.0) * 100 for step in acc_matrix[start_task:]]
            tasks = list(range(start_task + 1, len(class_order) + 1))
            plt.plot(tasks, accs, marker='o', label=f'Class {c}')
            
        # Plot mean seen accuracy
        mean_accs = [np.mean([step[c] for c in class_order[:i+1]]) * 100 for i, step in enumerate(acc_matrix)]
        plt.plot(range(1, 11), mean_accs, 'k--', linewidth=2, label='Mean Seen Acc')
        
        plt.title('TAGI-SNN Continual Learning: Accuracy over Tasks')
        plt.xlabel('Task (Class Number Learned)')
        plt.ylabel('Accuracy (%)')
        plt.xticks(range(1, 11))
        plt.ylim(-5, 105)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        out_png = os.path.join(plots_dir, 'continual_learning_results.png')
        plt.savefig(out_png, dpi=300)
        print(f"Saved plot to {out_png}")
    except ImportError:
        print("matplotlib not installed, skipping plot generation.")

if __name__ == "__main__":
    main()
