import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as AT
from scipy import ndimage
from tqdm import tqdm


class BirdSongPreprocessor:
    def __init__(self):
        # Key parameters from the paper:
        self.sample_rate = 32000  # Competition data is 32kHz
        self.n_fft = 1024  # FFT window size (~32ms at 32kHz)
        self.hop_length = 256  # 25% overlap as mentioned in BirdNET paper
        self.f_min = 150  # Min frequency 150 Hz
        self.f_max = 15000  # Max frequency 15 kHz
        self.n_mels = 64  # 64 mel bands

        # Initialize mel spectrogram transformer
        self.mel_spectrogram = AT.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            mel_scale="htk",  # Using HTK-style mel scaling
            power=2.0,  # Power spectrogram
            normalized=True,
            norm="slaney",  # Slaney-style mel normalization
        )

    def extract_signal_segments(
        self, waveform, threshold_factor=3.0, noise_threshold_factor=2.5
    ):
        """
        Extract segments containing bird vocalizations based on signal strength
        Implementation based on Sprengel et al., 2016 approach

        Args:
            waveform: Input audio waveform
            threshold_factor: Factor for signal detection (3.0 as in paper)
            noise_threshold_factor: Factor for noise detection (2.5 as in paper)

        Returns:
            signal_mask: Boolean mask indicating signal segments
            noise_mask: Boolean mask indicating noise segments
        """
        # Convert to spectrogram without log scaling for signal detection
        spec = self.mel_spectrogram(waveform)

        # Normalize spectrogram to [0,1] range as described in the paper
        spec_norm = spec / (torch.max(spec) + 1e-9)

        # Calculate row (frequency) and column (time) medians
        freq_medians = torch.median(spec_norm, dim=2, keepdim=True).values
        time_medians = torch.median(spec_norm, dim=1, keepdim=True).values

        # Select pixels that are N times bigger than both medians
        # For signal parts (threshold_factor = 3.0)
        signal_pixels = (spec_norm > (threshold_factor * freq_medians)) & (
            spec_norm > (threshold_factor * time_medians)
        )

        # For noise parts (noise_threshold_factor = 2.5)
        noise_pixels = (
            (spec_norm > (noise_threshold_factor * freq_medians))
            & (spec_norm > (noise_threshold_factor * time_medians))
            & ~signal_pixels
        )  # Ensure no overlap with signal

        # Apply binary erosion and dilation to clean up the masks
        # First convert to numpy for morphological operations
        signal_pixels_np = signal_pixels[0].cpu().numpy()
        noise_pixels_np = noise_pixels[0].cpu().numpy()

        # Define kernel for morphological operations (4x4 as in paper)
        kernel = np.ones((4, 4), np.uint8)

        # Apply erosion followed by dilation (opening operation)
        signal_pixels_np = ndimage.binary_erosion(signal_pixels_np, structure=kernel)
        signal_pixels_np = ndimage.binary_dilation(signal_pixels_np, structure=kernel)

        noise_pixels_np = ndimage.binary_erosion(noise_pixels_np, structure=kernel)
        noise_pixels_np = ndimage.binary_dilation(noise_pixels_np, structure=kernel)

        # Create indicator vectors (1 if column contains at least one 1)
        signal_indicator = np.any(signal_pixels_np, axis=0).astype(np.uint8)
        noise_indicator = np.any(noise_pixels_np, axis=0).astype(np.uint8)

        # Apply dilation to smooth the indicator vectors
        dilation_kernel = np.ones(4)
        signal_indicator = ndimage.binary_dilation(
            signal_indicator, structure=dilation_kernel
        )
        signal_indicator = ndimage.binary_dilation(
            signal_indicator, structure=dilation_kernel
        )

        noise_indicator = ndimage.binary_dilation(
            noise_indicator, structure=dilation_kernel
        )
        noise_indicator = ndimage.binary_dilation(
            noise_indicator, structure=dilation_kernel
        )

        # Ensure no overlap between signal and noise
        noise_indicator = noise_indicator & ~signal_indicator

        # Convert back to torch tensors
        signal_mask = torch.from_numpy(signal_indicator).to(waveform.device)
        noise_mask = torch.from_numpy(noise_indicator).to(waveform.device)

        # Scale masks to match waveform length
        # Calculate scaling factor
        spec_time_bins = signal_mask.shape[0]
        waveform_length = waveform.shape[1]
        scaling_factor = waveform_length / spec_time_bins

        # Create full-length masks
        full_signal_mask = torch.zeros(
            waveform_length, device=waveform.device, dtype=torch.bool
        )
        full_noise_mask = torch.zeros(
            waveform_length, device=waveform.device, dtype=torch.bool
        )

        # Map each spectrogram time bin to corresponding audio samples
        for i in range(spec_time_bins):
            start_idx = int(i * scaling_factor)
            end_idx = int((i + 1) * scaling_factor)
            if signal_mask[i]:
                full_signal_mask[start_idx:end_idx] = True
            if noise_mask[i]:
                full_noise_mask[start_idx:end_idx] = True

        return full_signal_mask, full_noise_mask

    def separate_signal_noise(self, waveform):
        """
        Separate audio into signal (bird vocalization) and noise parts

        Args:
            waveform: Input audio waveform

        Returns:
            signal_waveform: Audio containing only bird vocalizations
            noise_waveform: Audio containing only background noise
        """
        # Get signal and noise masks
        signal_mask, noise_mask = self.extract_signal_segments(waveform)

        # Create signal and noise waveforms
        signal_waveform = torch.zeros_like(waveform)
        noise_waveform = torch.zeros_like(waveform)

        # Apply masks
        signal_waveform[:, signal_mask] = waveform[:, signal_mask]
        noise_waveform[:, noise_mask] = waveform[:, noise_mask]

        return signal_waveform, noise_waveform

    def process_audio(self, audio_path, chunk_duration=3.0, overlap=0.5):
        """
        Process audio file into mel spectrograms, cutting into equal-sized chunks as
        described in Sprengel et al., using 3-second chunks as recommended in Kahl et al.

        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each chunk in seconds (default: 3.0s)
            overlap: Overlap between chunks as a fraction (default: 0.5 = 50%)

        Returns:
            signal_chunks: List of spectrograms from signal parts
            noise_chunks: List of spectrograms from noise parts
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = AT.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Separate signal and noise
        signal_waveform, noise_waveform = self.separate_signal_noise(waveform)

        # Calculate chunk parameters
        chunk_samples = int(chunk_duration * self.sample_rate)
        hop_samples = int(chunk_samples * (1 - overlap))

        # Process signal chunks
        signal_chunks = []
        for start in range(
            0, signal_waveform.shape[1] - chunk_samples + 1, hop_samples
        ):
            # Extract chunk
            chunk = signal_waveform[:, start : start + chunk_samples]

            # Skip chunks with no signal
            if torch.sum(chunk) > 0:
                # Convert to spectrogram
                spec = self.mel_spectrogram(chunk)
                spec = torch.log(spec + 1e-9)  # Log scaling
                signal_chunks.append(spec)

        # Process noise chunks
        noise_chunks = []
        for start in range(0, noise_waveform.shape[1] - chunk_samples + 1, hop_samples):
            # Extract chunk
            chunk = noise_waveform[:, start : start + chunk_samples]

            # Skip chunks with no noise
            if torch.sum(chunk) > 0:
                # Convert to spectrogram
                spec = self.mel_spectrogram(chunk)
                spec = torch.log(spec + 1e-9)  # Log scaling
                noise_chunks.append(spec)

        return signal_chunks, noise_chunks

    def augment_spectrogram(self, spec, noise_specs=None):
        """
        Apply domain-specific augmentations to spectrograms as described in BirdNET paper

        Args:
            spec (torch.Tensor): Input spectrogram
            noise_specs (list): List of noise spectrograms from non-salient chunks
        """
        # Maximum of three augmentations per sample as mentioned in the paper
        num_augmentations = np.random.randint(1, 4)
        augmented = spec.clone()

        # List of possible augmentations
        augmentations = [
            self._frequency_shift,
            self._time_shift,
            self._spec_warp,
            lambda x: self._add_ambient_noise(x, noise_specs) if noise_specs else x,
        ]

        # Randomly select and apply augmentations
        selected_augs = np.random.choice(
            augmentations,
            size=min(num_augmentations, len(augmentations)),
            replace=False,
        )

        for aug in selected_augs:
            if np.random.random() > 0.5:  # 0.5 probability as mentioned in paper
                augmented = aug(augmented)

        return augmented

    def _frequency_shift(self, spec, max_shift=10):
        """Vertical roll - Shift in frequency domain"""
        shift = np.random.randint(-max_shift, max_shift)
        return torch.roll(spec, shifts=shift, dims=1)

    def _time_shift(self, spec, max_shift=50):
        """Horizontal roll - Shift in time domain"""
        shift = np.random.randint(-max_shift, max_shift)
        return torch.roll(spec, shifts=shift, dims=2)

    def _spec_warp(self, spec):
        """
        Spectrogram warping similar to SpecAugment
        Applies random partial stretching in time and frequency
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

    def _add_ambient_noise(self, spec, noise_specs, max_weight=0.5):
        """
        Augment signal spectrogram with noise as described in Sprengel et al.

        Args:
            spec: Signal spectrogram to augment
            noise_specs: List of noise spectrograms to choose from
            max_weight: Maximum weight for noise addition

        Returns:
            Augmented spectrogram
        """
        if not noise_specs:
            return spec

        # Randomly select a noise spectrogram
        noise_spec = noise_specs[np.random.randint(len(noise_specs))]

        # Ensure shapes match
        if noise_spec.shape != spec.shape:
            # Resize noise spectrogram to match signal shape
            _, freq_dim, time_dim = spec.shape

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

        # Random weighting for noise (as in paper)
        weight = np.random.uniform(0, max_weight)

        # Add weighted noise
        augmented = (1 - weight) * spec + weight * noise_spec

        return augmented

    def collect_ambient_noise(self, audio_path):
        """
        Collect non-salient chunks for ambient noise augmentation
        Using the improved signal/noise separation method
        """
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            resampler = AT.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Separate signal and noise using our new method
        _, noise_waveform = self.separate_signal_noise(waveform)

        # Check if we have any noise segments
        if torch.sum(noise_waveform) > 0:
            # Convert to spectrogram
            noise_spec = self.mel_spectrogram(noise_waveform)
            noise_spec = torch.log(noise_spec + 1e-9)
            return noise_spec

        return None


def prepare_batch(
    audio_files, save_dir="train_audio_processed", training=True, show_progress=True
):
    """
    Prepare a batch of audio files for model training or inference.

    Args:
        audio_files (list): List of audio file paths
        save_dir (str): Directory to save the processed audio files
        training (bool): Whether to apply augmentation
        show_progress (bool): Whether to show progress bars
    """
    # Create save_dir if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata file to store mapping
    metadata = []
    signal_specs = []
    noise_specs = []

    # Initialize preprocessor
    preprocessor = BirdSongPreprocessor()

    # Group files by folder for better progress tracking
    files_by_folder = {}
    for file in audio_files:
        folder = os.path.basename(os.path.dirname(file))
        if folder not in files_by_folder:
            files_by_folder[folder] = []
        files_by_folder[folder].append(file)

    # Process audio files
    folder_iter = tqdm(
        files_by_folder.items(),
        desc="Processing folders",
        disable=not show_progress,
    )

    for folder, folder_files in folder_iter:
        # Create folder if it doesn't exist
        (save_dir / folder).mkdir(exist_ok=True)

        for audio_file in tqdm(
            folder_files,
            desc=f"Processing {folder}",
            leave=False,
            disable=not show_progress,
        ):
            # Create unique filenames for the processed specs
            base_filename = Path(audio_file).stem
            signal_dir = save_dir / folder / "signal"
            noise_dir = save_dir / folder / "noise"

            signal_dir.mkdir(exist_ok=True)
            noise_dir.mkdir(exist_ok=True)

            # Check if already processed
            signal_pattern = str(signal_dir / f"{base_filename}_*.pt")
            existing_signal_files = glob.glob(signal_pattern)

            if existing_signal_files:
                # Load existing spectrograms
                for file in existing_signal_files:
                    spec = torch.load(file)
                    signal_specs.append(spec)
                    metadata.append(
                        {
                            "original_file": audio_file,
                            "processed_file": file,
                            "folder": folder,
                            "type": "signal",
                        }
                    )

                # Load existing noise specs if available
                noise_pattern = str(noise_dir / f"{base_filename}_*.pt")
                for file in glob.glob(noise_pattern):
                    spec = torch.load(file)
                    noise_specs.append(spec)
                    metadata.append(
                        {
                            "original_file": audio_file,
                            "processed_file": file,
                            "folder": folder,
                            "type": "noise",
                        }
                    )

                continue

            try:
                # Process audio file into chunks
                signal_chunks, noise_chunks = preprocessor.process_audio(audio_file)

                # Augment signal chunks if in training mode
                if training:
                    signal_chunks = [
                        preprocessor.augment_spectrogram(chunk, noise_chunks)
                        for chunk in signal_chunks
                    ]

                # Save signal chunks
                for i, chunk in enumerate(signal_chunks):
                    chunk_file = signal_dir / f"{base_filename}_{i:03d}.pt"
                    torch.save(chunk, chunk_file)
                    signal_specs.append(chunk)
                    metadata.append(
                        {
                            "original_file": audio_file,
                            "processed_file": str(chunk_file),
                            "folder": folder,
                            "type": "signal",
                            "chunk_index": i,
                        }
                    )

                # Save noise chunks
                for i, chunk in enumerate(noise_chunks):
                    chunk_file = noise_dir / f"{base_filename}_{i:03d}.pt"
                    torch.save(chunk, chunk_file)
                    noise_specs.append(chunk)
                    metadata.append(
                        {
                            "original_file": audio_file,
                            "processed_file": str(chunk_file),
                            "folder": folder,
                            "type": "noise",
                            "chunk_index": i,
                        }
                    )

            except Exception as e:
                print(f"\nError processing {audio_file}: {str(e)}")
                continue

    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(save_dir / "metadata.csv", index=False)

    # Print summary
    print("\nProcessing Summary:")
    print(f"Total signal chunks: {len(signal_specs)}")
    print(f"Total noise chunks: {len(noise_specs)}")
    print("Files per folder:")
    folder_counts = metadata_df[metadata_df["type"] == "signal"][
        "folder"
    ].value_counts()
    print(folder_counts.head().to_string())

    return signal_specs, noise_specs, metadata_df


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess bird audio files")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/train_audio",
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="train_audio_processed",
        help="Directory to save processed files",
    )
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable augmentation (for validation/test data)",
    )

    args = parser.parse_args()

    # Get all .ogg files recursively
    audio_files = glob.glob(f"{args.input_dir}/**/*.ogg", recursive=True)
    print(f"Found {len(audio_files)} audio files")

    # Process files
    prepare_batch(audio_files, save_dir=args.output_dir, training=not args.no_augment)
