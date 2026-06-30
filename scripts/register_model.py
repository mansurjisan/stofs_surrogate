#!/usr/bin/env python3
"""Register an existing checkpoint in the MLflow Model Registry with lineage.

Registers an already-trained model without retraining. The model config is read from the
checkpoint (key ``config``) when present, otherwise from ``--config``. The registry needs a
database-backed tracking store (the file store does not support the registry):

    python scripts/register_model.py --checkpoint outputs/checkpoints/best_model.pt \\
        --tracking-uri sqlite:///mlflow.db --name stofs-gnn-midatlantic --alias staging
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stofs_surrogate.inference.predictor import Predictor
from stofs_surrogate.training.registry import DEFAULT_MODEL_NAME, ModelRegistry


def _load_config(checkpoint, config_path):
    if config_path:
        text = Path(config_path).read_text()
        if config_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(text)
        return json.loads(text)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        return checkpoint["config"]
    return {}


def main():
    p = argparse.ArgumentParser(description="Register a checkpoint with lineage")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db",
                   help="DB-backed MLflow store (the registry needs a DB, not the file store)")
    p.add_argument("--name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--model-type", default="stofs_gnn")
    p.add_argument("--model-kwargs", default=None,
                   help='JSON model constructor kwargs matching the checkpoint, e.g. '
                        '\'{"state_dim":1,"hidden_dim":128,"num_layers":6}\'')
    p.add_argument("--config", default=None, help="Config YAML/JSON, if not in the checkpoint")
    p.add_argument("--alias", default=None, help="Optional alias, e.g. staging or production")
    p.add_argument("--data-dir", default=None,
                   help="Directory of training files, for the data-manifest hash")
    args = p.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = _load_config(checkpoint, args.config)

    metrics = {}
    if isinstance(checkpoint, dict):
        for key in ("val_loss", "best_val_loss"):
            if isinstance(checkpoint.get(key), (int, float)):
                metrics[key] = float(checkpoint[key])

    # TODO(user): the production checkpoints use the inline PhysicsInformedCWLModel; to
    #   register those, import that class and pass a built `model=` to Predictor instead of
    #   relying on the package model_type + matching model_kwargs.
    model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
    predictor = Predictor.from_checkpoint(args.checkpoint, model_type=args.model_type,
                                          model_kwargs=model_kwargs)

    data_paths = None
    if args.data_dir:
        data_paths = [str(x) for x in Path(args.data_dir).glob("*")]

    registry = ModelRegistry(args.tracking_uri)
    version = registry.register(predictor.model, name=args.name, config=config,
                                metrics=metrics, data_paths=data_paths, alias=args.alias)
    print(f"Registered '{args.name}' version {version.version} "
          f"(git_sha + config + metrics logged as lineage).")


if __name__ == "__main__":
    main()
