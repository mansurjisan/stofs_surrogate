#!/usr/bin/env python3
"""CPU synthetic smoke-train: exercises the training + experiment-tracking plumbing.

Runs a tiny SimpleMeshGraphNet for a couple of epochs on synthetic SWE data, logging to
MLflow by default (writes to ./mlruns). Use this to verify tracking end-to-end without a
GPU or NOAA data.

    python scripts/smoke_train.py                  # MLflow -> ./mlruns
    python scripts/smoke_train.py --tracker none   # no tracking
    python scripts/smoke_train.py --tracker wandb  # Weights & Biases (if installed)
    mlflow ui                                      # then browse the "stofs-smoke" experiment
"""
import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stofs_surrogate.data.dataset import SyntheticSWEDataset
from stofs_surrogate.models.gnn import create_model
from stofs_surrogate.training.tracking import make_tracker
from stofs_surrogate.training.trainer import Trainer


def main():
    p = argparse.ArgumentParser(description="CPU synthetic smoke-train with experiment tracking")
    p.add_argument("--tracker", default="mlflow", choices=["mlflow", "wandb", "none"])
    p.add_argument("--tracking-uri", default=None,
                   help="MLflow tracking URI (default: ./mlruns)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--num-nodes", type=int, default=400)
    p.add_argument("--num-samples", type=int, default=40)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--output-dir", default="outputs/smoke")
    args = p.parse_args()

    config = {
        "data": {"type": "synthetic", "num_nodes": args.num_nodes,
                 "num_samples": args.num_samples, "include_velocity": True},
        "model": {"type": "simple", "hidden_dim": args.hidden_dim, "num_layers": 2},
        "training": {"batch_size": 4, "num_epochs": args.epochs,
                     "learning_rate": 1e-3, "device": "cpu"},
    }

    dataset = SyntheticSWEDataset(num_nodes=args.num_nodes, num_samples=args.num_samples,
                                  include_velocity=True)
    n_val = max(1, int(0.2 * len(dataset)))
    train_ds = torch.utils.data.Subset(dataset, range(len(dataset) - n_val))
    val_ds = torch.utils.data.Subset(dataset, range(len(dataset) - n_val, len(dataset)))

    train_loader = PyGDataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=4, shuffle=False)

    model = create_model("simple", input_dim=3, output_dim=3,
                         hidden_dim=args.hidden_dim, num_layers=2)

    tracker = make_tracker(args.tracker, experiment="stofs-smoke",
                           tracking_uri=args.tracking_uri)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        eval_every=1,
        save_every=args.epochs,
        mixed_precision=False,
        tracker=tracker,
        config=config,
    )

    history = trainer.train()
    print(f"Smoke training complete: {len(history['train_losses'])} epochs, "
          f"final train loss {history['train_losses'][-1]:.6f}")
    if args.tracker == "mlflow":
        uri = args.tracking_uri or "./mlruns"
        print(f"Logged to MLflow at {uri} (experiment 'stofs-smoke'). Run `mlflow ui` to view.")


if __name__ == "__main__":
    main()
