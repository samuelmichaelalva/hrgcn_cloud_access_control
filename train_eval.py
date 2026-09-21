import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from core.hrgcn import HRGCN
from core.augmentation import HetGDA
from data.cloud_dataset import CloudGraphGenerator


def train_single_model(
    ablation: str = "full",
    num_epochs: int = 20,
    lr: float = 0.008,
    alpha: float = 0.5,
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    generator = CloudGraphGenerator(feature_dim=8)
    train_graphs, test_graphs = generator.create_dataset(num_normal=200, num_attack=80)
    augmentor = HetGDA(num_node_types=4, num_edge_types=5)

    model = HRGCN(in_dim=8, hidden_dim=32, out_dim=16, ablation=ablation)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    bce_loss = nn.BCELoss()

    # Step 1: Initialize SVDD center c on initial normal representations
    model.eval()
    with torch.no_grad():
        init_embeddings = []
        for g in train_graphs[:40]:
            emb, _ = model.forward_graph_embedding(g.node_features, g.node_types, g.edge_index, g.edge_types)
            init_embeddings.append(emb)
        init_embeddings = torch.cat(init_embeddings, dim=0)
        model.init_svdd_center(init_embeddings)

    # Step 2: Training loop with gradient accumulation
    model.train()
    accum_steps = 8
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        for i, g in enumerate(train_graphs):
            # Normal graph forward pass
            g_emb, pred_normal, _ = model(g.node_features, g.node_types, g.edge_index, g.edge_types)
            loss_svdd = torch.sum((g_emb - model.svdd_center) ** 2)

            # Augmented graph (pseudo-anomaly) forward pass
            aug_x, aug_types, aug_e_idx, aug_e_types = augmentor.augment(
                g.node_features, g.node_types, g.edge_index, g.edge_types
            )
            _, pred_aug, _ = model(aug_x, aug_types, aug_e_idx, aug_e_types)

            # Self-Supervised BCE loss: Normal -> 0, Augmented -> 1
            loss_ss = bce_loss(pred_normal, torch.zeros_like(pred_normal)) + \
                      bce_loss(pred_aug, torch.ones_like(pred_aug))

            loss = (loss_svdd + alpha * loss_ss) / accum_steps
            loss.backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_graphs):
                optimizer.step()
                optimizer.zero_grad()

    # Step 3: Evaluation on test set
    model.eval()
    y_true = []
    y_scores = []
    latencies = []

    with torch.no_grad():
        for g in test_graphs:
            t0 = time.perf_counter()
            score_dict = model.compute_anomaly_score(
                g.node_features, g.node_types, g.edge_index, g.edge_types, beta=0.5
            )
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

            y_true.append(g.label)
            y_scores.append(score_dict["total_score"])

    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    avg_latency = float(np.mean(latencies))

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    return {
        "model": model,
        "ablation": ablation,
        "auc": float(auc),
        "ap": float(ap),
        "avg_latency_ms": round(avg_latency, 3),
        "roc_curve": {"fpr": [round(float(x), 4) for x in fpr.tolist()][::2], "tpr": [round(float(x), 4) for x in tpr.tolist()][::2]},
        "pr_curve": {"precision": [round(float(x), 4) for x in precision.tolist()][::2], "recall": [round(float(x), 4) for x in recall.tolist()][::2]}
    }


def run_benchmarks_and_train():
    print("=" * 60, flush=True)
    print("Training Full HRGCN and Running Ablation Benchmarks...", flush=True)
    print("=" * 60, flush=True)

    ablations = [
        ("full", "Full HRGCN (Hierarchical Relation-Augmented)"),
        ("no-edge-relation", "HRGCN-SDR (Source-Dest Hierarchy Only, No Edge Relation)"),
        ("no-node-relation", "HRGCN-ER (Edge Relation Only, No Node Hierarchy)"),
        ("vanilla-hetgcn", "Baseline HetGCN (Flat Heterogeneous Graph Convolution)")
    ]

    results = {}
    best_model = None

    for mode, label in ablations:
        print(f"\n[+] Training mode: {label} ...", flush=True)
        t_start = time.time()
        res = train_single_model(ablation=mode, num_epochs=20)
        duration = time.time() - t_start

        results[mode] = {
            "name": label,
            "auc": round(res["auc"] * 100, 2),
            "ap": round(res["ap"] * 100, 2),
            "latency_ms": res["avg_latency_ms"],
            "train_duration_s": round(duration, 2),
            "roc_curve": res["roc_curve"],
            "pr_curve": res["pr_curve"]
        }

        print(f"    --> ROC-AUC: {results[mode]['auc']}% | AP: {results[mode]['ap']}% | Latency: {results[mode]['latency_ms']} ms", flush=True)

        if mode == "full":
            best_model = res["model"]

    # Save best model checkpoint
    os.makedirs("models", exist_ok=True)
    checkpoint_path = os.path.join("models", "hrgcn_cloud_model.pt")
    torch.save({
        "state_dict": best_model.state_dict(),
        "svdd_center": best_model.svdd_center,
        "in_dim": 8,
        "hidden_dim": 32,
        "out_dim": 16,
        "num_node_types": 4,
        "num_edge_types": 5,
        "ablation": "full"
    }, checkpoint_path)
    print(f"\n[OK] Saved trained HRGCN checkpoint to: {checkpoint_path}", flush=True)

    # Save metrics JSON
    metrics_path = os.path.join("models", "benchmark_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved benchmark metrics to: {metrics_path}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    run_benchmarks_and_train()
