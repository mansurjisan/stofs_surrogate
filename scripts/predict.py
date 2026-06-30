#!/usr/bin/env python3
"""Run a forecast with the package Predictor (the package-native inference path).

With ``--checkpoint`` it loads a trained package model; otherwise it builds a small random
STOFSSurrogateGNN for a shape demo. This is the thin-wrapper pattern the production
rollout scripts should eventually adopt (once their inline model is promoted to the package).

    python scripts/predict.py --num-steps 12
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stofs_surrogate.inference.predictor import Predictor
from stofs_surrogate.models.gnn import create_model


def main():
    p = argparse.ArgumentParser(description="Package-native rollout demo")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--num-nodes", type=int, default=200)
    p.add_argument("--num-steps", type=int, default=12)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.manual_seed(0)
    num_edges = args.num_nodes * 4
    state = torch.randn(args.num_nodes, 3)
    node_features = torch.randn(args.num_nodes, 3)
    edge_index = torch.randint(0, args.num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 3)

    if args.checkpoint:
        predictor = Predictor.from_checkpoint(args.checkpoint, model_type="stofs_gnn",
                                              device=args.device)
    else:
        model = create_model("stofs_gnn", state_dim=3, node_feature_dim=3,
                             edge_feature_dim=3, hidden_dim=32, num_layers=2)
        predictor = Predictor(model, device=args.device)

    preds = predictor.rollout(state, node_features, edge_index, edge_attr,
                              num_steps=args.num_steps)
    print(f"Rollout output shape: {tuple(preds.shape)}  "
          f"(expected [{args.num_steps + 1}, {args.num_nodes}, 3])")


if __name__ == "__main__":
    main()
