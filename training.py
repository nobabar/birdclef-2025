import ast
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
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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


class SqueezeExcitation(nn.Module):
    """Squeeze-Excitation block for channel attention."""

    def __init__(self, channels, reduction_ratio=16):
        """Initialize the Squeeze-Excitation block."""
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        squeeze_channels = max(1, channels // reduction_ratio)
        self.fc1 = nn.Conv2d(
            channels, squeeze_channels, 1
        )  # Using Conv2d instead of Linear for shape handling
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass for the Squeeze-Excitation block."""
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale


class InvertedResidual(nn.Module):
    """Inverted Residual block with SE (MBConv)."""

    def __init__(
        self, in_channels, out_channels, stride=1, expand_ratio=6, dropout_prob=0.5
    ):
        """Initialize the Inverted Residual block."""
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        exp_channels = in_channels * expand_ratio
        self.conv = nn.Sequential(
            # Expansion
            nn.Conv2d(in_channels, exp_channels, 1, bias=False),
            nn.BatchNorm2d(exp_channels),
            nn.SiLU(),  # SiLU/Swish activation as used in EfficientNet
            # Depthwise
            nn.Conv2d(
                exp_channels,
                exp_channels,
                3,
                stride,
                1,
                groups=exp_channels,
                bias=False,
            ),
            nn.BatchNorm2d(exp_channels),
            nn.SiLU(),
            # Squeeze-Excitation
            SqueezeExcitation(exp_channels),
            # Projection
            nn.Conv2d(exp_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=dropout_prob),
        )

    def forward(self, x):
        """Forward pass for the Inverted Residual block."""
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class BirdNETv2(nn.Module):
    """BirdNET-V2 implementation with EfficientNet-like architecture."""

    def __init__(self, num_classes, dropout_prob=0.5):
        """Initialize the BirdNETv2 model."""
        super().__init__()

        # Initial processing
        self.conv_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )

        # EfficientNet-like backbone (simplified version)
        self.blocks = nn.Sequential(
            InvertedResidual(32, 64, stride=1, dropout_prob=dropout_prob),
            InvertedResidual(64, 64, stride=1, dropout_prob=dropout_prob),
            InvertedResidual(64, 128, stride=2, dropout_prob=dropout_prob),
            InvertedResidual(128, 128, stride=1, dropout_prob=dropout_prob),
            InvertedResidual(128, 256, stride=2, dropout_prob=dropout_prob),
            InvertedResidual(256, 256, stride=1, dropout_prob=dropout_prob),
            InvertedResidual(256, 512, stride=2, dropout_prob=dropout_prob),
            InvertedResidual(512, 512, stride=1, dropout_prob=dropout_prob),
        )

        # Final layers
        self.conv_head = nn.Sequential(
            nn.Conv2d(512, 1024, 1, bias=False), nn.BatchNorm2d(1024), nn.SiLU()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        """Forward pass for the BirdNETv2 model."""
        # Input shape: [batch_size, 1, height, width]
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


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

        # Get all unique classes from both primary and secondary labels
        primary_labels = set(self.metadata["train_primary_label"].unique())
        # Convert string representation of list to actual list and get unique values
        secondary_labels = set()
        for labels in self.metadata["train_secondary_labels"].dropna():
            labels_list = ast.literal_eval(labels)
            secondary_labels.update(labels_list)

        # Combine and sort all unique classes
        self.classes = sorted(primary_labels.union(secondary_labels))
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

        # Get file info from metadata
        row = self.metadata.iloc[idx]
        file_path = row["processed_file"]

        # Create multi-label tensor
        labels = torch.zeros(len(self.classes), dtype=torch.float32)

        # Set primary label
        primary_idx = self.class_to_idx[row["train_primary_label"]]
        labels[primary_idx] = 1.0

        # Set secondary labels if they exist
        if pd.notna(row["train_secondary_labels"]):
            secondary_list = ast.literal_eval(row["train_secondary_labels"])
            for label in secondary_list:
                if label in self.class_to_idx:
                    sec_idx = self.class_to_idx[label]
                    labels[sec_idx] = 1.0

        try:
            spectrogram = torch.load(file_path)
            # Ensure spectrogram is float32
            spectrogram = spectrogram.to(torch.float32)

            if self.transform:
                spectrogram = self.transform(spectrogram)

            # Return weight as float32
            return spectrogram, labels, torch.tensor(1.0, dtype=torch.float32)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a zero spectrogram of the expected shape if loading fails
            # Return zero spectrogram and weight as float32
            return (
                torch.zeros(1, 64, 384, dtype=torch.float32),
                labels,
                torch.tensor(0.0, dtype=torch.float32),
            )


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
        dropout_prob=0.5,
        mixup=True,
        dataset=None,
    ):
        """Initialize the BirdNETLightning module."""
        super().__init__()

        # Model
        self.model = BirdNETv2(
            num_classes=num_classes,
            dropout_prob=dropout_prob,
        )

        # Learning rate
        self.learning_rate = learning_rate

        # Mixup flag
        self.mixup = mixup
        self.dataset = dataset

        # Use BCEWithLogitsLoss for multi-label classification
        self.criterion = nn.BCEWithLogitsLoss()

        # Save hyperparameters
        self.save_hyperparameters(ignore=["dataset"])

    def forward(self, x):
        """Forward pass for the BirdNETLightning module."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step for the BirdNETLightning module."""
        # Unpack batch with metadata
        spectrograms, labels, weights = batch

        # Apply mixup if enabled
        if self.mixup and self.dataset is not None:
            spectrograms, labels = self.apply_mixup(spectrograms, labels)

        # Continue with forward pass and loss calculation
        outputs = self(spectrograms)
        loss = self.criterion(outputs, labels)

        # Add more detailed logging
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_output_mean", outputs.mean(), on_step=False, on_epoch=True)
        self.log("train_output_std", outputs.std(), on_step=False, on_epoch=True)
        self.log(
            "train_labels_mean", labels.float().mean(), on_step=False, on_epoch=True
        )

        # Print some debugging info occasionally
        if batch_idx % 100 == 0:
            print(f"\nTraining batch {batch_idx}:")
            print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
            print(f"Labels range: [{labels.min():.3f}, {labels.max():.3f}]")
            print(f"Loss: {loss.item():.3f}")

        return loss

    def validation_step(self, batch, batch_idx):
        """Perform validation step for the BirdNETLightning module."""
        spectrograms, labels, weights = batch
        outputs = self(spectrograms)
        loss = self.criterion(outputs, labels)

        # Calculate mAP only if there are positive labels
        predictions = outputs.cpu().numpy()
        targets = labels.cpu().numpy()

        # Check if there are any positive labels
        if targets.sum() > 0:
            mAP = average_precision_score(targets, predictions, average="macro")
        else:
            mAP = 0.0  # or some other appropriate value
            print(f"Warning: Batch {batch_idx} has no positive labels")

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

    # Add some debug info
    print("Total number of classes:", len(full_dataset.classes))
    print("Total number of samples:", len(full_dataset))

    # If validation directory is provided, use it and set val_split to 0.0
    if val_data_dir:
        val_dataset = BirdSongDataset(
            processed_dir=val_data_dir,
            transform=None,
            augment=False,
        )
        val_split = 0.0

    # Create stratified splits
    dataset_size = len(full_dataset)

    # Get all labels to use for stratification
    all_labels = []
    label_counts = {}

    # First pass: count samples per label
    print("\nAnalyzing dataset distribution...")
    for i in tqdm(range(dataset_size), desc="Counting samples per class"):
        _, labels, _ = full_dataset[i]
        positive_indices = torch.where(labels == 1)[0]
        if len(positive_indices) > 0:
            strat_label = positive_indices[0].item()
            label_counts[strat_label] = label_counts.get(strat_label, 0) + 1

    # Find labels with enough samples for stratification (at least 3 samples)
    valid_labels = {label for label, count in label_counts.items() if count >= 3}

    # After calculating valid_labels:
    print("\nClass distribution:")
    sorted_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    for label, count in sorted_counts[:10]:  # Print top 10 most common classes
        species_name = full_dataset.classes[label]
        print(f"Species {species_name}: {count} samples")
    print("...")
    for label, count in sorted_counts[-10:]:  # Print bottom 10 least common classes
        species_name = full_dataset.classes[label]
        print(f"Species {species_name}: {count} samples")

    # Second pass: assign stratification labels
    for i in tqdm(range(dataset_size), desc="Assigning stratification labels"):
        _, labels, _ = full_dataset[i]
        positive_indices = torch.where(labels == 1)[0]

        # Use first valid label, or 0 if none found
        strat_label = 0
        if len(positive_indices) > 0:
            for idx in positive_indices:
                if idx.item() in valid_labels:
                    strat_label = idx.item()
                    break
        all_labels.append(strat_label)

    # Print some statistics
    print(f"\nFound {len(valid_labels)} classes with ≥3 samples for stratification")
    print(f"Min samples per class: {min(label_counts.values())}")
    print(f"Max samples per class: {max(label_counts.values())}")

    # Create stratified split
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_split + val_split, random_state=42
    )

    # Get train and temp indices
    train_idx, temp_idx = next(sss.split(np.zeros(len(all_labels)), all_labels))

    # Use simple random split when stratification fails
    if val_split > 0:
        val_test_labels = [all_labels[i] for i in temp_idx]
        try:
            sss_val = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_split / (test_split + val_split),
                random_state=42,
            )
            val_idx_temp, test_idx_temp = next(
                sss_val.split(np.zeros(len(temp_idx)), val_test_labels)
            )
        except ValueError:
            print(
                "\nWarning: Falling back to random split for validation/test due to insufficient samples"
            )
            # Use random split instead
            n_val = int(len(temp_idx) * val_split / (val_split + test_split))
            val_idx_temp = np.arange(n_val)
            test_idx_temp = np.arange(n_val, len(temp_idx))

        val_idx = temp_idx[val_idx_temp]
        test_idx = temp_idx[test_idx_temp]
    else:
        test_idx = temp_idx
        val_idx = []

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = (
        torch.utils.data.Subset(full_dataset, val_idx) if val_split > 0 else None
    )
    test_dataset = (
        torch.utils.data.Subset(full_dataset, test_idx) if test_split > 0 else None
    )

    print(
        f"Dataset split: {len(train_idx)} train ({len(train_idx)/dataset_size:.1%}), "
        f"{len(val_idx)} validation ({len(val_idx)/dataset_size:.1%}), "
        f"{len(test_idx)} test ({len(test_idx)/dataset_size:.1%})"
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
        "--max_epochs", type=int, default=3, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Initial learning rate"
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
        dropout_prob=args.dropout_prob,
        mixup=not args.no_mixup,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        plot_curves=args.plot_curves,
        log_dir=args.log_dir,
    )

    print(f"Best model saved at: {best_model_path}")
