"""
Training pipeline for STOFS surrogate model.

Features:
- Mixed precision training
- Gradient accumulation for large graphs
- Learning rate scheduling
- Checkpointing and early stopping
- Logging to TensorBoard/W&B
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Callable
import logging
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for STOFS surrogate GNN.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        device: str = 'cuda',
        output_dir: str = 'outputs',
        # Training settings
        num_epochs: int = 100,
        grad_clip: float = 1.0,
        grad_accumulation_steps: int = 1,
        mixed_precision: bool = True,
        # Checkpointing
        save_every: int = 10,
        eval_every: int = 5,
        early_stopping_patience: int = 20,
        # Logging
        log_every: int = 10,
        use_wandb: bool = False,
        wandb_project: str = 'stofs-surrogate',
        wandb_run_name: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: The GNN model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (default: AdamW)
            scheduler: LR scheduler (default: CosineAnnealingLR)
            criterion: Loss function (default: MSELoss)
            device: Device to train on
            output_dir: Directory for outputs
            num_epochs: Number of training epochs
            grad_clip: Gradient clipping norm
            grad_accumulation_steps: Steps to accumulate gradients
            mixed_precision: Whether to use AMP
            save_every: Save checkpoint every N epochs
            eval_every: Evaluate every N epochs
            early_stopping_patience: Stop if no improvement for N evals
            log_every: Log metrics every N steps
            use_wandb: Whether to use W&B logging
            wandb_project: W&B project name
            wandb_run_name: W&B run name
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.grad_accumulation_steps = grad_accumulation_steps
        self.save_every = save_every
        self.eval_every = eval_every
        self.early_stopping_patience = early_stopping_patience
        self.log_every = log_every
        self.use_wandb = use_wandb

        # Create output directories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=1e-5,
            )
        else:
            self.optimizer = optimizer

        # Setup scheduler
        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=1e-6,
            )
        else:
            self.scheduler = scheduler

        # Setup criterion
        self.criterion = criterion or nn.MSELoss()

        # Setup mixed precision
        self.mixed_precision = mixed_precision and device == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.mixed_precision else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

        # Setup W&B
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        'model': type(model).__name__,
                        'num_params': sum(p.numel() for p in model.parameters()),
                        'num_epochs': num_epochs,
                        'grad_clip': grad_clip,
                        'mixed_precision': mixed_precision,
                    }
                )
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed, disabling W&B logging")
                self.use_wandb = False
                self.wandb = None

        # Setup TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            logger.warning("tensorboard not available")
            self.writer = None

    def train(self) -> Dict[str, List[float]]:
        """
        Run full training loop.

        Returns:
            Dictionary of training history
        """
        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Device: {self.device}, Mixed precision: {self.mixed_precision}")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_loss = self._train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = None
            if self.val_loader is not None and (epoch + 1) % self.eval_every == 0:
                val_loss = self._validate()
                self.val_losses.append(val_loss)

                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    self._save_checkpoint('best_model.pt')
                else:
                    self.epochs_without_improvement += 1

                # Early stopping
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            # Learning rate step
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

            # Log progress
            lr = self.optimizer.param_groups[0]['lr']
            msg = f"Epoch {epoch + 1:3d}/{self.num_epochs} | Train Loss: {train_loss:.6f}"
            if val_loss is not None:
                msg += f" | Val Loss: {val_loss:.6f}"
            msg += f" | LR: {lr:.2e}"
            logger.info(msg)

            # W&B logging
            if self.use_wandb:
                log_dict = {'train_loss': train_loss, 'lr': lr, 'epoch': epoch}
                if val_loss is not None:
                    log_dict['val_loss'] = val_loss
                self.wandb.log(log_dict)

        # Training complete
        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed/60:.1f} minutes")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")

        # Save final model
        self._save_checkpoint('final_model.pt')

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = batch.to(self.device)

            # Forward pass with optional mixed precision
            if self.mixed_precision:
                with torch.amp.autocast('cuda'):
                    loss = self._compute_loss(batch)
                    loss = loss / self.grad_accumulation_steps

                self.scaler.scale(loss).backward()
            else:
                loss = self._compute_loss(batch)
                loss = loss / self.grad_accumulation_steps
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.grad_accumulation_steps
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item() * self.grad_accumulation_steps:.6f}'})

            # TensorBoard logging
            if self.writer is not None and self.global_step % self.log_every == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)

        # Flush any remaining accumulated gradients (tail step)
        if (batch_idx + 1) % self.grad_accumulation_steps != 0:
            if self.mixed_precision:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1

        return total_loss / num_batches

    def _validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

                if self.mixed_precision:
                    with torch.amp.autocast('cuda'):
                        loss = self._compute_loss(batch)
                else:
                    loss = self._compute_loss(batch)

                total_loss += loss.item()
                num_batches += 1

        val_loss = total_loss / num_batches

        if self.writer is not None:
            self.writer.add_scalar('val/loss', val_loss, self.global_step)

        return val_loss

    def _compute_loss(self, batch) -> torch.Tensor:
        """Compute loss for a batch."""
        # Get model prediction
        if hasattr(self.model, 'forward_pyg'):
            pred = self.model.forward_pyg(batch)
        else:
            # Fallback for simpler models
            pred = self.model(
                batch.x,
                batch.pos,
                batch.edge_index,
                batch.depth if hasattr(batch, 'depth') else batch.node_features[:, 2],
            )

        # Compute loss
        loss = self.criterion(pred, batch.y)

        return loss

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']

        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


def train_stofs_surrogate(
    model: nn.Module,
    train_dataset,
    val_dataset=None,
    batch_size: int = 4,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    device: str = 'cuda',
    output_dir: str = 'outputs',
    **trainer_kwargs,
) -> Dict[str, List[float]]:
    """
    Convenience function to train STOFS surrogate model.

    Args:
        model: The GNN model
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on
        output_dir: Output directory
        **trainer_kwargs: Additional arguments for Trainer

    Returns:
        Training history
    """
    # Create data loaders
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set > 0 for multi-process loading
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = PyGDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        output_dir=output_dir,
        num_epochs=num_epochs,
        **trainer_kwargs,
    )

    # Train
    history = trainer.train()

    return history
