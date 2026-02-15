#!/usr/bin/env python3
"""
Train STOFS surrogate on real ADCIRC/STOFS output.

Usage:
    python scripts/train_stofs.py --config config/config.yaml

Requirements:
    - Place fort.14 mesh file in data/raw/
    - Place fort.63.nc (elevation) in data/raw/
    - Optionally place fort.64.nc (velocity) in data/raw/
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import torch
import yaml
import logging
from torch.utils.data import random_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from mesh import ADCIRCMesh, create_mesh
from dataset import ADCIRCDataset, SyntheticSWEDataset, MultiEventDataset
from model import STOFSSurrogateGNN, SimpleMeshGraphNet, create_model
from trainer import train_stofs_surrogate


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train STOFS surrogate model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--mesh', type=str, default=None,
                        help='Override mesh path')
    parser.add_argument('--elevation', type=str, default=None,
                        help='Override elevation path')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (cuda/cpu)')
    args = parser.parse_args()

    # Load config
    project_root = Path(__file__).parent.parent
    config_path = project_root / args.config
    config = load_config(config_path)

    # Override from command line
    if args.mesh:
        config['data']['mesh_path'] = args.mesh
    if args.elevation:
        config['data']['elevation_path'] = args.elevation
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.device:
        config['hardware']['device'] = args.device

    print("=" * 60)
    print("STOFS Surrogate Model Training")
    print("=" * 60)

    # Device setup
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'

    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Memory: {mem_gb:.1f} GB")

    # Create dataset
    print("\n1. Loading data...")

    if config['data']['use_synthetic']:
        # Use synthetic data
        syn_config = config['data']['synthetic']
        dataset = SyntheticSWEDataset(
            num_nodes=syn_config['num_nodes'],
            num_samples=syn_config['num_samples'],
            domain_size=tuple(syn_config['domain_size']),
            include_velocity=syn_config['include_velocity'],
            include_forcing=syn_config['include_forcing'],
        )
        state_dim = 3 if syn_config['include_velocity'] else 1
        forcing_dim = 3 if syn_config['include_forcing'] else 0

    elif config['multi_event']['enabled']:
        # Multi-event training
        mesh_path = project_root / config['data']['mesh_path']
        mesh = create_mesh(str(mesh_path))

        event_paths = []
        for event in config['multi_event']['events']:
            event_paths.append({
                'elevation_path': str(project_root / event['elevation_path']),
                'velocity_path': str(project_root / event.get('velocity_path', '')),
            })

        dataset = MultiEventDataset(
            mesh=mesh,
            event_paths=event_paths,
            normalize=config['data']['normalize'],
            eta_scale=config['data']['eta_scale'],
            vel_scale=config['data']['vel_scale'],
        )
        state_dim = 3  # eta, u, v
        forcing_dim = 0

    else:
        # Single event training
        mesh_path = project_root / config['data']['mesh_path']
        elev_path = project_root / config['data']['elevation_path']
        vel_path = config['data'].get('velocity_path')
        if vel_path:
            vel_path = project_root / vel_path

        # Check files exist
        if not mesh_path.exists():
            logger.error(f"Mesh file not found: {mesh_path}")
            logger.info("Please place your fort.14 file in data/raw/")
            logger.info("Or use synthetic data: set data.use_synthetic=true in config")
            sys.exit(1)

        if not elev_path.exists():
            logger.error(f"Elevation file not found: {elev_path}")
            logger.info("Please place your fort.63.nc file in data/raw/")
            sys.exit(1)

        # Load mesh
        print(f"   Loading mesh: {mesh_path}")
        mesh = create_mesh(str(mesh_path))

        # Check if mesh needs subsampling (for 4GB GPU)
        if device == 'cuda' and mesh.num_nodes > 50000:
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if mem_gb < 8:
                target_nodes = 30000
                logger.warning(f"Large mesh ({mesh.num_nodes:,} nodes) with limited GPU memory")
                logger.warning(f"Subsampling to {target_nodes:,} nodes")
                mesh = mesh.subsample(target_nodes, method='farthest')

        # Load forcing if enabled
        forcing_paths = None
        if config['data']['forcing']['enabled']:
            forcing_paths = {
                'wind_u': str(project_root / config['data']['forcing']['wind_u_path']),
                'wind_v': str(project_root / config['data']['forcing']['wind_v_path']),
                'pressure': str(project_root / config['data']['forcing']['pressure_path']),
            }
            forcing_dim = 3
        else:
            forcing_dim = 0

        # Create dataset
        dataset = ADCIRCDataset(
            mesh=mesh,
            elevation_path=str(elev_path),
            velocity_path=str(vel_path) if vel_path and vel_path.exists() else None,
            forcing_paths=forcing_paths,
            time_stride=config['data']['time_stride'],
            normalize=config['data']['normalize'],
            eta_scale=config['data']['eta_scale'],
            vel_scale=config['data']['vel_scale'],
            cache_data=config['data']['cache_data'],
        )

        # Determine state dimension
        sample = dataset[0]
        state_dim = sample.x.shape[1]

    print(f"   Dataset size: {len(dataset)} samples")

    # Train/val split
    train_ratio = config['data']['train_ratio']
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Get dimensions
    sample = dataset[0]
    node_feature_dim = sample.node_features.shape[1]
    edge_feature_dim = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 3

    print(f"   State dim: {state_dim}")
    print(f"   Node feature dim: {node_feature_dim}")
    print(f"   Forcing dim: {forcing_dim}")

    # Create model
    print("\n2. Creating model...")

    model_type = config['model']['type']

    if model_type == 'simple':
        model_config = config['model']['simple']
        model = SimpleMeshGraphNet(
            input_dim=state_dim,
            output_dim=state_dim,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
        )
    elif model_type == 'stofs_gnn':
        model_config = config['model']['stofs_gnn']
        model = STOFSSurrogateGNN(
            state_dim=state_dim,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            forcing_dim=forcing_dim,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            activation=model_config['activation'],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: {model_type}")
    print(f"   Parameters: {num_params:,}")

    # Train
    print("\n3. Training...")

    train_config = config['training']
    output_dir = project_root / config['output']['dir']

    history = train_stofs_surrogate(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=train_config['batch_size'],
        num_epochs=train_config['num_epochs'],
        learning_rate=train_config['learning_rate'],
        device=device,
        output_dir=str(output_dir),
        save_every=train_config['save_every'],
        eval_every=train_config['eval_every'],
        early_stopping_patience=train_config['early_stopping_patience'],
        mixed_precision=train_config['mixed_precision'] and device == 'cuda',
        log_every=train_config['log_every'],
        grad_clip=train_config['grad_clip'],
        use_wandb=config['output']['use_wandb'],
        wandb_project=config['output']['wandb_project'],
    )

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal train loss: {history['train_losses'][-1]:.6f}")
    if history['val_losses']:
        print(f"Best val loss: {min(history['val_losses']):.6f}")

    print(f"\nModel saved to: {output_dir}/checkpoints/")
    print("\nNext steps for ensemble generation:")
    print("  1. Load trained model")
    print("  2. Perturb input forcing (wind, IC)")
    print("  3. Run GNN inference for each perturbation")
    print("  4. Aggregate ensemble statistics")


if __name__ == '__main__':
    main()
