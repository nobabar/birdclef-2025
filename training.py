import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""

    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.5):
        """Initialize the basic residual block."""
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        """Forward pass for the basic block."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DownsamplingBlock(nn.Module):
    """Downsampling residual block with strided convolution."""

    def __init__(self, in_channels, out_channels, dropout_prob=0.5):
        """Initialize the downsampling block."""
        super(DownsamplingBlock, self).__init__()
        # As per Xie et al., 2018 suggestion for downsampling blocks
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)

        # Shortcut with strided convolution
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        """Forward pass for the downsampling block."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class WideResNet(nn.Module):
    """Wide ResNet implementation based on BirdNET paper."""

    def __init__(self, num_classes, width_factor=4, depth_factor=3, dropout_prob=0.5):
        """Initialize the Wide ResNet model."""
        super(WideResNet, self).__init__()

        # Initial number of channels (scaled by width factor K)
        self.init_channels = 16 * width_factor

        # Pre-processing block
        self.preprocessing = nn.Sequential(
            nn.Conv2d(
                1, self.init_channels, kernel_size=5, stride=1, padding=2, bias=False
            ),
            nn.BatchNorm2d(self.init_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # Pooling in time dimension only
        )

        # Residual stacks
        # First stack (no downsampling in first block)
        self.stack1 = self._make_stack(
            self.init_channels, self.init_channels, depth_factor, 1, dropout_prob
        )

        # Second stack (with downsampling)
        self.stack2 = self._make_stack(
            self.init_channels, 2 * self.init_channels, depth_factor, 2, dropout_prob
        )

        # Third stack (with downsampling)
        self.stack3 = nn.Sequential(
            DownsamplingBlock(
                2 * self.init_channels, 4 * self.init_channels, dropout_prob
            ),
            *[
                BasicBlock(
                    4 * self.init_channels, 4 * self.init_channels, 1, dropout_prob
                )
                for _ in range(depth_factor - 1)
            ],
        )

        # Classification block (as per Schlüter, 2018)
        # Assuming input is 64x384, after preprocessing and stacks it becomes:
        # 64x192 -> 32x96 -> 16x48
        # So the final feature map size is (batch_size, 4*init_channels, 16, 48)

        # 1x1 convolution to reduce channels
        self.conv_reduce = nn.Conv2d(
            4 * self.init_channels, 512, kernel_size=1, bias=False
        )
        self.bn_reduce = nn.BatchNorm2d(512)

        # Global pooling and classification
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)

    def _make_stack(self, in_channels, out_channels, num_blocks, stride, dropout_prob):
        layers = []
        # First block may have downsampling
        if stride == 2:
            layers.append(DownsamplingBlock(in_channels, out_channels, dropout_prob))
        else:
            layers.append(BasicBlock(in_channels, out_channels, stride, dropout_prob))

        # Rest of the blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1, dropout_prob))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass for the Wide ResNet model."""
        # Input shape: [batch_size, 1, 64, 384] (mel spectrogram)

        # Pre-processing
        out = self.preprocessing(x)  # [batch_size, init_channels, 64, 192]

        # Residual stacks
        out = self.stack1(out)  # [batch_size, init_channels, 64, 192]
        out = self.stack2(out)  # [batch_size, 2*init_channels, 32, 96]
        out = self.stack3(out)  # [batch_size, 4*init_channels, 16, 48]

        # Classification block
        out = F.relu(self.bn_reduce(self.conv_reduce(out)))  # [batch_size, 512, 16, 48]
        out = self.classifier(out)  # [batch_size, num_classes, 16, 48]

        # Global log-mean-exponential pooling (as described in the paper)
        # This produces 3 predictions per 3-second spectrogram (1 per second)
        out = torch.exp(out)
        out = torch.mean(out, dim=(2, 3))  # Global average pooling
        out = torch.log(out + 1e-7)  # Log to stabilize

        # Apply sigmoid activation for multi-label classification
        out = torch.sigmoid(out)

        return out


class BirdSongDataset(Dataset):
    """Dataset for loading processed bird song spectrograms."""

    def __init__(self, processed_dir, transform=None, augment=False):
        """
        Initialize the dataset.

        Args:
            processed_dir: Directory with all the processed spectrograms and metadata
            transform: Optional transform to be applied on a sample
            augment: Whether to use augmented samples

        """
        self.processed_dir = Path(processed_dir)
        self.transform = transform
        self.augment = augment

        # Load metadata from the processed directory
        metadata_path = self.processed_dir / "metadata.csv"
        self.metadata = pd.read_csv(metadata_path)

        # Filter for signal spectrograms only (not noise)
        self.metadata = self.metadata[self.metadata["type"] == "signal"]

        # Get unique classes (folders)
        self.classes = sorted(self.metadata["folder"].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        print(
            f"Loaded {len(self.metadata)} signal chunks across {len(self.classes)} classes"
        )

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx):
        """Get an item from the dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get file path directly from metadata
        file_path = self.metadata.iloc[idx]["processed_file"]

        # Load spectrogram
        try:
            spectrogram = torch.load(file_path)
            # Ensure spectrogram is float32
            spectrogram = spectrogram.float()
        except FileNotFoundError:
            # If the path in metadata is relative, try to resolve it
            file_path = self.processed_dir / file_path
            spectrogram = torch.load(file_path).float()

        # Get label (folder name is the class)
        folder = self.metadata.iloc[idx]["folder"]
        label_idx = self.class_to_idx[folder]

        # Convert to one-hot encoding (ensure float32)
        label = torch.zeros(len(self.classes), dtype=torch.float32)
        label[label_idx] = 1.0

        # Get sample weight based on metadata
        weight = torch.tensor(1.0, dtype=torch.float32)

        # Adjust weight based on recording quality
        if "train_rating" in self.metadata.columns and not pd.isna(
            self.metadata.iloc[idx]["train_rating"]
        ):
            rating = float(self.metadata.iloc[idx]["train_rating"])
            # Higher rated recordings get higher weights
            weight *= 1.0 + rating / 5.0

        # Adjust weight for underrepresented collections
        if "train_collection" in self.metadata.columns:
            collection = self.metadata.iloc[idx]["train_collection"]
            if collection == "iNat":  # If iNaturalist is underrepresented
                weight *= 1.2

        # Apply transforms if any
        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label, weight


class MixupTransform:
    """Mixup augmentation for spectrograms."""

    def __init__(self, dataset, num_mix=3, alpha=0.2):
        """
        Initialize the MixupTransform.

        Args:
            dataset: The dataset to sample from
            num_mix: Maximum number of samples to mix
            alpha: Parameter for beta distribution

        """
        self.dataset = dataset
        self.num_mix = num_mix
        self.alpha = alpha

    def __call__(self, spectrogram, label):
        """
        Apply mixup to the spectrogram and label.

        Args:
            spectrogram: Input spectrogram
            label: One-hot encoded label

        Returns:
            Mixed spectrogram and label

        """
        # Determine number of samples to mix (1-3)
        num_to_mix = np.random.randint(1, self.num_mix + 1)

        # Start with the original sample
        mixed_spec = spectrogram.clone()
        mixed_label = label.clone()

        # Mix with random samples
        for _ in range(num_to_mix - 1):
            # Sample random index
            idx = np.random.randint(0, len(self.dataset))

            # Get random sample
            random_spec, random_label = self.dataset[idx]

            # Sample mixing weight from beta distribution
            lam = np.random.beta(self.alpha, self.alpha)

            # Mix spectrogram and label
            mixed_spec = lam * mixed_spec + (1 - lam) * random_spec
            mixed_label = lam * mixed_label + (1 - lam) * random_label

        return mixed_spec, mixed_label


class BirdNETLightning(pl.LightningModule):
    """PyTorch Lightning module for BirdNET."""

    def __init__(
        self,
        num_classes,
        learning_rate=1e-3,
        width_factor=4,
        depth_factor=3,
        dropout_prob=0.5,
        mixup=True,
        dataset=None,
    ):
        """Initialize the BirdNETLightning module."""
        super(BirdNETLightning, self).__init__()
        self.save_hyperparameters()

        # Model
        self.model = WideResNet(
            num_classes=num_classes,
            width_factor=width_factor,
            depth_factor=depth_factor,
            dropout_prob=dropout_prob,
        )

        # Learning rate
        self.learning_rate = learning_rate

        # Mixup flag
        self.mixup = mixup

        # Binary cross-entropy loss for multi-label classification
        self.criterion = nn.BCELoss()

        # Dataset for mixup
        self.dataset = dataset

    def forward(self, x):
        """Forward pass for the BirdNETLightning module."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step for the BirdNETLightning module."""
        # Unpack batch with metadata
        spectrograms, labels, weights = batch

        # Apply adaptive mixup based on metadata
        if self.mixup:
            # Create a default mask (all False)
            batch_size = spectrograms.shape[0]
            high_quality_mask = torch.zeros(
                batch_size, dtype=torch.bool, device=spectrograms.device
            )

            # If weights > 1.8, consider it high quality (rating >= 4 would give weight >= 1.8)
            # This is based on the weight calculation: weight *= 1.0 + rating / 5.0
            high_quality_mask = weights > 1.8

            # Apply different mixup strategies based on quality
            spectrograms_mixed = torch.zeros_like(spectrograms)
            labels_mixed = torch.zeros_like(labels)

            # Process high quality recordings if any exist
            if high_quality_mask.any():
                # High quality recordings get gentler mixup
                high_quality_specs = spectrograms[high_quality_mask]
                high_quality_labels = labels[high_quality_mask]

                # Apply mixup with lower alpha (less aggressive)
                mixed_specs, mixed_labels = self.apply_mixup(
                    high_quality_specs,
                    high_quality_labels,
                    alpha=0.2,  # Less aggressive
                )
                spectrograms_mixed[high_quality_mask] = mixed_specs
                labels_mixed[high_quality_mask] = mixed_labels

            # Process lower quality recordings if any exist
            if (~high_quality_mask).any():
                # Lower quality recordings get more aggressive mixup
                low_quality_specs = spectrograms[~high_quality_mask]
                low_quality_labels = labels[~high_quality_mask]

                # Apply mixup with higher alpha (more aggressive)
                mixed_specs, mixed_labels = self.apply_mixup(
                    low_quality_specs,
                    low_quality_labels,
                    alpha=0.4,  # More aggressive
                )
                spectrograms_mixed[~high_quality_mask] = mixed_specs
                labels_mixed[~high_quality_mask] = mixed_labels

            spectrograms = spectrograms_mixed
            labels = labels_mixed

        # Continue with forward pass and loss calculation
        outputs = self(spectrograms)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Perform validation step for the BirdNETLightning module."""
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Calculate mAP (mean Average Precision)
        y_np = y.cpu().numpy()
        y_hat_np = y_hat.cpu().numpy()

        # Handle case where a class has no positive samples in the batch
        with np.errstate(divide="ignore", invalid="ignore"):
            ap_scores = [
                average_precision_score(y_np[:, i], y_hat_np[:, i])
                if np.sum(y_np[:, i]) > 0
                else np.nan
                for i in range(y_np.shape[1])
            ]

        # Filter out NaN values
        ap_scores = [score for score in ap_scores if not np.isnan(score)]
        mAP = np.mean(ap_scores) if ap_scores else 0.0

        # Log validation metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mAP", mAP, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": loss, "val_mAP": mAP}

    def test_step(self, batch, batch_idx):
        """Test step for the BirdNETLightning module."""
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Calculate mAP
        y_np = y.cpu().numpy()
        y_hat_np = y_hat.cpu().numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            ap_scores = [
                average_precision_score(y_np[:, i], y_hat_np[:, i])
                if np.sum(y_np[:, i]) > 0
                else np.nan
                for i in range(y_np.shape[1])
            ]

        ap_scores = [score for score in ap_scores if not np.isnan(score)]
        mAP = np.mean(ap_scores) if ap_scores else 0.0

        # Log test metrics
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_mAP", mAP, on_epoch=True)

        return {"test_loss": loss, "test_mAP": mAP}

    def configure_optimizers(self):
        """Configure the optimizers for the BirdNETLightning module."""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Learning rate scheduler with step-wise reduction
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }

    def on_validation_epoch_end(self):
        """Analyze performance by metadata categories."""
        # Reduce dropout probability by 0.1 when learning rate is reduced
        # This is mentioned in the paper
        if hasattr(self, "last_lr") and self.last_lr != self.learning_rate:
            # Learning rate was reduced
            for module in self.modules():
                if isinstance(module, nn.Dropout2d):
                    module.p = max(0.0, module.p - 0.1)

            self.last_lr = self.learning_rate
        else:
            self.last_lr = self.learning_rate

        # Analyze performance by metadata categories
        if hasattr(self, "val_metadata"):
            # Group results by collection
            collection_metrics = {}
            for collection in self.val_metadata["train_collection"].unique():
                mask = self.val_metadata["train_collection"] == collection
                collection_preds = self.val_preds[mask]
                collection_targets = self.val_targets[mask]

                # Calculate mAP for this collection
                collection_map = average_precision_score(
                    collection_targets.cpu().numpy(), collection_preds.cpu().numpy()
                )

                collection_metrics[collection] = collection_map
                self.log(f"val_mAP_{collection}", collection_map)

            # Log geographic performance
            # ... similar analysis by region ...

    def apply_mixup(self, spectrograms, labels, alpha=0.2):
        """
        Apply mixup augmentation to a batch of spectrograms and labels.

        Args:
            spectrograms: Batch of spectrograms [batch_size, channels, height, width]
            labels: Batch of one-hot encoded labels [batch_size, num_classes]
            alpha: Parameter for beta distribution (lower = less aggressive mixing)

        Returns:
            mixed_spectrograms: Mixed batch of spectrograms
            mixed_labels: Mixed batch of labels

        """
        batch_size = spectrograms.shape[0]

        # If batch size is 1, can't mix
        if batch_size <= 1:
            return spectrograms, labels

        # Sample mixing weights from beta distribution
        weights = (
            torch.from_numpy(np.random.beta(alpha, alpha, size=batch_size))
            .float()
            .to(spectrograms.device)
        )

        # Expand weights for broadcasting
        weights = weights.view(-1, 1, 1, 1)

        # Create random permutation of the batch
        indices = torch.randperm(batch_size).to(spectrograms.device)

        # Mix spectrograms
        mixed_spectrograms = (
            weights * spectrograms + (1 - weights) * spectrograms[indices]
        )

        # Adjust weights for labels (remove extra dimensions)
        label_weights = weights.view(-1, 1)

        # Mix labels
        mixed_labels = label_weights * labels + (1 - label_weights) * labels[indices]

        return mixed_spectrograms, mixed_labels


def plot_learning_curves(trainer, save_path):
    """
    Plot training and validation metrics.

    Args:
        trainer: PyTorch Lightning trainer instance
        save_path: Path to save the plot

    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Get metrics from trainer's logged metrics
    metrics = trainer.callback_metrics

    # Get training history from logged metrics
    train_losses = []
    val_losses = []
    val_maps = []

    # Access the metrics from the trainer's logged metrics
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            train_losses = callback.best_k_models.get("train_loss", [])
            val_losses = callback.best_k_models.get("val_loss", [])
            val_maps = callback.best_k_models.get("val_mAP", [])

    # Plot training and validation loss if we have data
    if train_losses and val_losses:
        epochs = range(len(train_losses))
        ax1.plot(epochs, train_losses, "b-", label="Training Loss")
        ax1.plot(epochs, val_losses, "r-", label="Validation Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

    # Plot validation mAP if we have data
    if val_maps:
        epochs = range(len(val_maps))
        ax2.plot(epochs, val_maps, "g-", label="Validation mAP")
        ax2.set_title("Validation mAP")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("mAP")
        ax2.legend()
        ax2.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Learning curves saved to {save_path}")


def train_birdnet(
    train_data_dir,
    val_data_dir=None,
    val_split=0.1,
    test_split=0.0,
    batch_size=32,
    max_epochs=10,
    learning_rate=1e-3,
    width_factor=4,
    depth_factor=3,
    dropout_prob=0.5,
    mixup=True,
    num_workers=4,
    checkpoint_dir="checkpoints",
    plot_curves=False,
    log_dir="lightning_logs",
):
    """
    Train the BirdNET model.

    Args:
        train_data_dir: Directory containing processed training data
        val_data_dir: Directory containing processed validation data (if None, use train_data_dir)
        val_split: Fraction of data to use for validation (0.0 to 1.0)
        test_split: Fraction of data to use for testing (0.0 to 1.0)
        batch_size: Batch size for training
        max_epochs: Maximum number of epochs to train
        learning_rate: Initial learning rate
        width_factor: Width scaling factor for Wide ResNet (K)
        depth_factor: Depth scaling factor for Wide ResNet (N)
        dropout_prob: Initial dropout probability
        mixup: Whether to use mixup augmentation
        num_workers: Number of workers for data loading
        checkpoint_dir: Directory to save checkpoints
        plot_curves: Whether to plot and save learning curves
        log_dir: Directory for TensorBoard logs

    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load the full dataset
    full_dataset = BirdSongDataset(
        processed_dir=train_data_dir,
        transform=None,
        augment=False,
    )

    # If validation directory is provided, use it and set val_split to 0.0
    if val_data_dir:
        val_dataset = BirdSongDataset(
            processed_dir=val_data_dir,
            transform=None,
            augment=False,
        )
        val_split = 0.0

    # Create datasets based on split configuration
    # Calculate split sizes
    dataset_size = len(full_dataset)
    test_size = int(dataset_size * test_split)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - test_size - val_size

    # Create random splits
    generator = torch.Generator().manual_seed(42)  # For reproducibility
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    print(
        f"Dataset split: {train_size} train ({train_size/dataset_size:.1%}), "
        f"{val_size} validation ({val_size/dataset_size:.1%}), "
        f"{test_size} test ({test_size/dataset_size:.1%})"
    )

    # Set augmentation for training set
    train_dataset.dataset.augment = mixup

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=torch.cuda.is_available(),
    )

    # Create validation loader if validation data exists
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=torch.cuda.is_available(),
        )

    # Create test loader if test data exists
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=torch.cuda.is_available(),
        )

    # Get number of classes
    num_classes = len(full_dataset.classes)
    print(f"Training with {num_classes} classes")

    # Create model
    model = BirdNETLightning(
        num_classes=num_classes,
        learning_rate=learning_rate,
        width_factor=width_factor,
        depth_factor=depth_factor,
        dropout_prob=dropout_prob,
        mixup=mixup,
        dataset=train_dataset if mixup else None,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="birdnet-{epoch:02d}-{val_mAP:.4f}",
        monitor="val_mAP",
        mode="max",
        save_top_k=3,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,  # With cooldown of 3 as mentioned in the paper
        mode="min",
        min_delta=0.001,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Create logger
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name="birdnet",  # This will create a subdirectory for your experiment
        version=None,  # Auto-increment version
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        accelerator="auto",  # Use GPU if available
        devices=1,
        log_every_n_steps=1,
        logger=logger,  # Use our configured logger
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Plot learning curves if requested
    if plot_curves:
        plot_dir = Path(checkpoint_dir) / "plots"
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / "learning_curves.png"
        plot_learning_curves(trainer, plot_path)

    # Test model if test data exists
    if test_loader:
        print("\nEvaluating model on test set...")
        test_results = trainer.test(model, test_loader, verbose=True)
        print(f"Test results: {test_results}")

    # Return best model path
    return checkpoint_callback.best_model_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train BirdNET model")
    parser.add_argument(
        "--train_dir",
        type=str,
        default="data/train_audio_processed",
        help="Directory containing processed training data",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=None,
        help="Directory containing processed validation data (if None, use train_dir)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,  # Default to 10% test split
        help="Fraction of data to use for validation (0.0 to 1.0, default: 0.1 for 80/10/10 split)",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,  # Default to 10% test split
        help="Fraction of data to use for testing (0.0 to 1.0, default: 0.1 for 80/10/10 split)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Initial learning rate"
    )
    parser.add_argument(
        "--width_factor",
        type=int,
        default=4,
        help="Width scaling factor for Wide ResNet (K)",
    )
    parser.add_argument(
        "--depth_factor",
        type=int,
        default=3,
        help="Depth scaling factor for Wide ResNet (N)",
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.5, help="Initial dropout probability"
    )
    parser.add_argument(
        "--no_mixup", action="store_true", help="Disable mixup augmentation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--plot_curves",
        action="store_true",
        help="Plot and save learning curves",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory for TensorBoard logs",
    )

    args = parser.parse_args()

    # Train model
    best_model_path = train_birdnet(
        train_data_dir=args.train_dir,
        val_data_dir=args.val_dir,
        val_split=args.val_split,
        test_split=args.test_split,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        width_factor=args.width_factor,
        depth_factor=args.depth_factor,
        dropout_prob=args.dropout_prob,
        mixup=not args.no_mixup,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        plot_curves=args.plot_curves,
        log_dir=args.log_dir,
    )

    print(f"Best model saved at: {best_model_path}")
