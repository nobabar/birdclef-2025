import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class BirdSongAugmenter:
    """Augmenter for bird song spectrograms."""

    def __init__(self, seed=42):
        """Initialize the augmenter with optional random seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def frequency_shift(self, spec, max_shift_pct=0.1):
        """
        Apply frequency shift augmentation.

        Args:
            spec: Input spectrogram (1, freq_bins, time_bins)
            max_shift_pct: Maximum shift as percentage of frequency bins

        Returns:
            Augmented spectrogram

        """
        _, freq_bins, _ = spec.shape
        max_shift = int(freq_bins * max_shift_pct)

        if max_shift == 0:
            return spec

        # Random shift amount (positive = shift up, negative = shift down)
        shift = np.random.randint(-max_shift, max_shift + 1)

        # Create shifted spectrogram
        shifted = torch.zeros_like(spec)

        if shift > 0:
            # Shift up (lose bottom frequencies)
            shifted[:, shift:, :] = spec[:, :-shift, :]
        elif shift < 0:
            # Shift down (lose top frequencies)
            shifted[:, :shift, :] = spec[:, -shift:, :]
        else:
            # No shift
            shifted = spec

        return shifted

    def time_shift(self, spec, max_shift_pct=0.2):
        """
        Apply time shift augmentation.

        Args:
            spec: Input spectrogram (1, freq_bins, time_bins)
            max_shift_pct: Maximum shift as percentage of time bins

        Returns:
            Augmented spectrogram

        """
        _, _, time_bins = spec.shape
        max_shift = int(time_bins * max_shift_pct)

        if max_shift == 0:
            return spec

        # Random shift amount
        shift = np.random.randint(-max_shift, max_shift + 1)

        # Create shifted spectrogram
        shifted = torch.zeros_like(spec)

        if shift > 0:
            # Shift right
            shifted[:, :, shift:] = spec[:, :, :-shift]
        elif shift < 0:
            # Shift left
            shifted[:, :, :shift] = spec[:, :, -shift:]
        else:
            # No shift
            shifted = spec

        return shifted

    def spec_warp(self, spec):
        """
        Spectrogram warping similar to SpecAugment.

        Applies random partial stretching in time and frequency.

        """
        freq_dim, time_dim = spec.shape[1:]

        # Create warping parameters
        w = np.random.randint(5, 20)  # window size
        center_freq = np.random.randint(w, freq_dim - w)
        center_time = np.random.randint(w, time_dim - w)

        # Create warping matrix
        factor = np.random.uniform(0.8, 1.2)
        warped = spec.clone()

        # Apply warping around center point
        warped[
            :, center_freq - w : center_freq + w, center_time - w : center_time + w
        ] *= factor

        return warped

    def add_noise(self, spec, noise_specs, max_weight=0.5):
        """
        Add random noise from a collection of noise spectrograms.

        Args:
            spec: Input spectrogram
            noise_specs: List of noise spectrograms
            max_weight: Maximum weight for noise addition

        Returns:
            Augmented spectrogram

        """
        if not noise_specs:
            return spec

        # Get target shape
        _, freq_dim, time_dim = spec.shape

        # Randomly select a noise spectrogram
        noise_spec = noise_specs[np.random.randint(len(noise_specs))]

        # Resize noise spectrogram to match target shape
        if noise_spec.shape[1:] != spec.shape[1:]:
            # Handle frequency dimension mismatch (shouldn't happen with same preprocessing)
            if noise_spec.shape[1] != freq_dim:
                # Interpolate frequency dimension
                noise_spec = torch.nn.functional.interpolate(
                    noise_spec, size=(freq_dim, noise_spec.shape[2]), mode="bilinear"
                )

            # Handle time dimension mismatch
            if noise_spec.shape[2] != time_dim:
                # Center crop or pad
                if noise_spec.shape[2] > time_dim:
                    # Center crop
                    start = (noise_spec.shape[2] - time_dim) // 2
                    noise_spec = noise_spec[:, :, start : start + time_dim]
                else:
                    # Pad
                    pad_size = time_dim - noise_spec.shape[2]
                    pad_left = pad_size // 2
                    pad_right = pad_size - pad_left
                    noise_spec = torch.nn.functional.pad(
                        noise_spec, (pad_left, pad_right)
                    )

        # Random weighting for noise
        weight = np.random.uniform(0, max_weight)

        # Add weighted noise
        augmented = (1 - weight) * spec + weight * noise_spec

        return augmented


def augment_dataset(
    processed_dir="train_audio_processed",
    output_dir=None,
    augmentations_per_sample=3,
    freq_shift_prob=0.5,
    time_shift_prob=0.5,
    spec_warp_prob=0.5,
    noise_prob=0.5,
    show_progress=True,
):
    """
    Augment a dataset of processed spectrograms.

    For each original spectrogram, create multiple augmented versions.
    Each augmented version can have up to 3 different augmentation techniques applied.

    Args:
        processed_dir: Directory containing processed spectrograms
        output_dir: Directory to save augmented spectrograms
        augmentations_per_sample: Number of augmented versions to create per sample
        freq_shift_prob: Probability of applying frequency shift to each augmented version
        time_shift_prob: Probability of applying time shift to each augmented version
        spec_warp_prob: Probability of applying spectrogram warping to each augmented version
        noise_prob: Probability of applying noise addition to each augmented version
        show_progress: Whether to show progress bars

    """
    # Create output directory
    if output_dir is None:
        output_dir = Path(processed_dir)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize augmenter
    augmenter = BirdSongAugmenter()

    # Load metadata
    metadata_path = Path(processed_dir) / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    # Get all signal and noise spectrograms
    signal_files = metadata[metadata["type"] == "signal"]["processed_file"].tolist()
    noise_files = metadata[metadata["type"] == "noise"]["processed_file"].tolist()

    # Load noise spectrograms for augmentation
    noise_specs = []
    if noise_files:
        # Sample a subset of noise files to save memory
        sampled_noise_files = random.sample(noise_files, min(1000, len(noise_files)))

        for file in tqdm(
            sampled_noise_files,
            desc="Loading noise spectrograms",
            disable=not show_progress,
        ):
            try:
                noise_spec = torch.load(file)
                noise_specs.append(noise_spec)
            except Exception as e:
                print(f"Error loading noise file {file}: {e}")

    # Process each signal file
    augmented_metadata = []

    for signal_file in tqdm(
        signal_files, desc="Augmenting spectrograms", disable=not show_progress
    ):
        try:
            # Load original spectrogram
            spec = torch.load(signal_file)

            # Get original metadata
            orig_meta = (
                metadata[metadata["processed_file"] == signal_file].iloc[0].to_dict()
            )

            # Create output directory structure
            species_folder = Path(signal_file).parent.name
            output_species_dir = output_dir / species_folder
            output_species_dir.mkdir(exist_ok=True)

            # Base filename
            base_filename = Path(signal_file).stem

            # Create multiple augmented versions of each sample
            for i in range(augmentations_per_sample):
                # Start with a copy of the original spectrogram
                augmented = spec.clone()

                # Track which augmentations were applied
                applied_augmentations = []

                # Determine how many augmentation techniques to apply (1-3)
                num_techniques = np.random.randint(1, 4)

                # Randomly select which augmentation techniques to apply
                # without replacement (don't apply the same technique twice)
                techniques = []
                if np.random.random() < freq_shift_prob:
                    techniques.append("freq_shift")
                if np.random.random() < time_shift_prob:
                    techniques.append("time_shift")
                if np.random.random() < spec_warp_prob:
                    techniques.append("spec_warp")
                if np.random.random() < noise_prob and noise_specs:
                    techniques.append("noise")

                # Limit to the number of techniques we want to apply
                if len(techniques) > num_techniques:
                    # Prioritize noise augmentation if it's in the list
                    if "noise" in techniques and num_techniques < len(techniques):
                        # Keep noise and randomly select from the others
                        techniques.remove("noise")
                        other_techniques = np.random.choice(
                            techniques, size=num_techniques - 1, replace=False
                        ).tolist()
                        techniques = ["noise"] + other_techniques
                    else:
                        # No noise or already removed, just random selection
                        techniques = np.random.choice(
                            techniques, size=num_techniques, replace=False
                        ).tolist()

                # Apply the selected techniques
                for technique in techniques:
                    if technique == "freq_shift":
                        augmented = augmenter.frequency_shift(augmented)
                        applied_augmentations.append("freq_shift")
                    elif technique == "time_shift":
                        augmented = augmenter.time_shift(augmented)
                        applied_augmentations.append("time_shift")
                    elif technique == "spec_warp":
                        augmented = augmenter.spec_warp(augmented)
                        applied_augmentations.append("spec_warp")
                    elif technique == "noise":
                        augmented = augmenter.add_noise(augmented, noise_specs)
                        applied_augmentations.append("noise")

                # Save augmented spectrogram with augmentation types in filename
                aug_types_str = "_".join(
                    [t[:4] for t in applied_augmentations]
                )  # Abbreviate for filename length
                aug_filename = f"{base_filename}_aug{i:02d}_{aug_types_str}.pt"
                aug_path = output_species_dir / aug_filename
                torch.save(augmented, aug_path)

                # Add to metadata
                aug_meta = orig_meta.copy()
                aug_meta["processed_file"] = str(aug_path)
                aug_meta["augmentation_id"] = i
                aug_meta["original_file"] = signal_file
                aug_meta["applied_augmentations"] = ",".join(applied_augmentations)
                augmented_metadata.append(aug_meta)

        except Exception as e:
            print(f"Error processing {signal_file}: {e}")
            continue

    # Save augmented metadata
    augmented_df = pd.DataFrame(augmented_metadata)
    augmented_df.to_csv(output_dir / "augmented_metadata.csv", index=False)

    # Print summary
    print("\nAugmentation Summary:")
    print(f"Original signal files: {len(signal_files)}")
    print(f"Augmented files created: {len(augmented_metadata)}")
    print(f"Augmentations per sample: {augmentations_per_sample}")

    # Print augmentation statistics
    if augmented_metadata:
        aug_counts = {}
        for meta in augmented_metadata:
            if "applied_augmentations" in meta:
                augs = meta["applied_augmentations"].split(",")
                for aug in augs:
                    aug_counts[aug] = aug_counts.get(aug, 0) + 1

        print("\nAugmentation Statistics:")
        for aug, count in aug_counts.items():
            print(f"{aug}: {count} ({count/len(augmented_metadata)*100:.1f}%)")

    return augmented_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Augment processed bird audio spectrograms"
    )
    parser.add_argument(
        "--input_dir",
        "-I",
        type=str,
        default="train_audio_processed",
        help="Directory containing processed spectrograms",
    )
    parser.add_argument(
        "--augmentations",
        "-A",
        type=int,
        default=3,
        help="Maximum number of augmented versions to create per sample",
    )
    parser.add_argument(
        "--freq_shift_prob",
        "-F",
        type=float,
        default=0.5,
        help="Probability for frequency shift augmentation",
    )
    parser.add_argument(
        "--time_shift_prob",
        "-T",
        type=float,
        default=0.5,
        help="Probability for time shift augmentation",
    )
    parser.add_argument(
        "--spec_warp_prob",
        "-W",
        type=float,
        default=0.5,
        help="Probability for spectrogram warping augmentation",
    )
    parser.add_argument(
        "--noise_prob",
        "-N",
        type=float,
        default=0.5,
        help="Probability for noise augmentation",
    )

    args = parser.parse_args()

    # Augment dataset
    augment_dataset(
        processed_dir=args.input_dir,
        augmentations_per_sample=args.augmentations,
        freq_shift_prob=args.freq_shift_prob,
        time_shift_prob=args.time_shift_prob,
        spec_warp_prob=args.spec_warp_prob,
        noise_prob=args.noise_prob,
    )
