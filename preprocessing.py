import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as AT
from scipy import ndimage
from silero_vad import get_speech_timestamps, load_silero_vad
from tqdm import tqdm

from waveform_comparaison import (
    compare_energy,
    visualize_waveform_in_seconds,
)

# from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H


class BirdSongPreprocessor:
    """Preprocess bird song audio files into mel spectrograms."""

    def __init__(self):
        """Initialize the BirdSongPreprocessor."""
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
        self.vad_model = load_silero_vad()

    def extract_signal_segments(
        self, waveform, threshold_factor=3.5, noise_threshold_factor=2.0
    ):
        """
        Extract segments containing bird vocalizations based on signal strength.

        Implementation based on Sprengel et al., 2016 approach.

        Args:
        ----
            waveform: Input audio waveform
            threshold_factor: Factor for signal detection (3.5 instead of 3.0 in paper)
            noise_threshold_factor: Factor for noise detection (2.0 instead of 2.5 in paper)

        Returns:
        -------
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
        # For signal parts (threshold_factor = 3.5)
        signal_pixels = (spec_norm > (threshold_factor * freq_medians)) & (
            spec_norm > (threshold_factor * time_medians)
        )

        # For noise parts (noise_threshold_factor = 2.0)
        noise_pixels = (spec_norm > (noise_threshold_factor * freq_medians)) & (
            spec_norm > (noise_threshold_factor * time_medians)
        )

        # Apply binary erosion and dilation to clean up the masks
        # First convert to numpy for morphological operations
        signal_pixels_np = signal_pixels[0].cpu().numpy()
        noise_pixels_np = noise_pixels[0].cpu().numpy()

        # Define kernel for morphological operations (4x4 as in paper)
        kernel = np.ones((4, 4), np.uint8)

        # Apply erosion followed by dilation (opening operation)
        signal_pixels_np = ndimage.binary_erosion(signal_pixels_np, structure=kernel)
        signal_pixels_np = ndimage.binary_dilation(signal_pixels_np, structure=kernel)

        # For noise, just apply erosion to shrink it (no dilation)
        noise_pixels_np = ndimage.binary_erosion(noise_pixels_np, structure=kernel)
        # noise_pixels_np = ndimage.binary_dilation(noise_pixels_np, structure=kernel)

        # Create indicator vectors (1 if column contains at least one 1)
        signal_indicator = np.any(signal_pixels_np, axis=0).astype(np.uint8)
        noise_indicator = np.any(noise_pixels_np, axis=0).astype(np.uint8)

        # Apply dilation to smooth the signal indicator vector
        dilation_kernel = np.ones(2)
        signal_indicator = ndimage.binary_dilation(
            signal_indicator, structure=dilation_kernel
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
        Separate audio into signal (bird vocalization) and noise parts.

        Args:
        ----
            waveform: Input audio waveform

        Returns:
        -------
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

    def remove_human_voice_using_silero(self, waveform):
        """
        Remove segments of waveform where human voice is detected using Silero VAD.

        Args:
        ----
            waveform: Tensor of shape (1, N)

        Returns:
        -------
            waveform with human voice segments zeroed out
        """
        # Silero VAD expects 16kHz mono audio
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.sample_rate != 16000:
            resampler = AT.Resample(orig_freq=self.sample_rate, new_freq=16000)
            waveform_16k = resampler(waveform)
        else:
            waveform_16k = waveform.clone()

        # Detect voice segments
        speech_timestamps = get_speech_timestamps(
            waveform_16k[0], self.vad_model, sampling_rate=16000
        )

        # Map detected speech timestamps back to original sample rate
        waveform_clean = waveform.clone()
        for segment in speech_timestamps:
            start = int(segment["start"] * self.sample_rate / 16000)
            end = int(segment["end"] * self.sample_rate / 16000)
            waveform_clean[:, start:end] = 0  # zero out human voice

        return waveform_clean

    def process_audio(
        self, audio_path, chunk_duration=3.0, overlap=0.5, visualization_dir=None
    ):
        """
        Process audio file into mel spectrograms.

        Process audio file into mel spectrograms, cutting into equal-sized chunks as
        described in Sprengel et al., using 3-second chunks as recommended in Kahl et al.

        Args:
        ----
            audio_path: Path to audio file
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks (0.0-1.0)

        Returns:
        -------
            signal_chunks: List of mel spectrograms containing bird vocalizations
            noise_chunks: List of mel spectrograms containing background noise

        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = AT.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Remove human voice
        original_waveform = waveform
        waveform = self.remove_human_voice_using_silero(waveform)

        # Compare energy before and after voice removal
        compare_energy(original_waveform, waveform)

        ######
        # Visualize the waveform
        ######
        if visualization_dir:
            filename = Path(audio_path).stem
            visualization_path = os.path.join(
                visualization_dir, f"{filename}_waveform.png"
            )
        else:
            visualization_path = None

        visualize_waveform_in_seconds(
            original_waveform, waveform, sr, save_path=visualization_path
        )

        # Separate signal and noise
        signal_waveform, noise_waveform = self.separate_signal_noise(waveform)

        # Calculate chunk size and hop size in samples
        chunk_size = int(chunk_duration * self.sample_rate)
        hop_size = int(chunk_size * (1 - overlap))

        # Calculate number of chunks
        num_chunks = max(1, int((waveform.shape[1] - chunk_size) / hop_size) + 1)

        signal_chunks = []
        noise_chunks = []

        # Process each chunk
        for i in range(num_chunks):
            # Calculate start and end indices
            start = i * hop_size
            end = start + chunk_size

            # If end is beyond the waveform, pad with zeros
            if end > waveform.shape[1]:
                # Create padded chunk
                signal_chunk = torch.zeros(1, chunk_size, device=waveform.device)
                noise_chunk = torch.zeros(1, chunk_size, device=waveform.device)

                # Copy available samples
                signal_chunk[:, : waveform.shape[1] - start] = signal_waveform[
                    :, start : waveform.shape[1]
                ]
                noise_chunk[:, : waveform.shape[1] - start] = noise_waveform[
                    :, start : waveform.shape[1]
                ]
            else:
                # Extract chunk
                signal_chunk = signal_waveform[:, start:end]
                noise_chunk = noise_waveform[:, start:end]

            # Check if signal chunk contains any signal
            if torch.sum(signal_chunk**2) > 0:
                # Convert to mel spectrogram
                spec = self.mel_spectrogram(signal_chunk)
                spec = torch.log(spec + 1e-9)  # Log scaling
                signal_chunks.append(spec)

            # Check if noise chunk contains any noise
            if torch.sum(noise_chunk**2) > 0:
                # Convert to mel spectrogram
                spec = self.mel_spectrogram(noise_chunk)
                spec = torch.log(spec + 1e-9)  # Log scaling
                noise_chunks.append(spec)

        return signal_chunks, noise_chunks


def prepare_batch(
    audio_files,
    metadata_path="data/train.csv",
    save_dir="train_audio_processed",
    show_progress=True,
):
    """
    Prepare a batch of audio files for model training or inference.

    Args:
    ----
        audio_files (list): List of audio file paths
        metadata_path (str): Path to the train.csv file with additional metadata
        save_dir (str): Directory to save the processed audio files
        show_progress (bool): Whether to show progress bars

    """
    # Create save_dir if it doesn't exist
    save_dir = Path("data", save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load train.csv metadata if available
    train_metadata = {}
    if metadata_path and os.path.exists(metadata_path):
        try:
            train_df = pd.read_csv(metadata_path)
            for _, row in train_df.iterrows():
                train_metadata[row["filename"]] = row.to_dict()
            print(f"Loaded metadata for {len(train_metadata)} files from train.csv")
        except Exception as e:
            print(f"Error loading train.csv metadata: {e}")

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

            # Get filename for metadata lookup
            rel_path = os.path.relpath(audio_file, "data/train_audio_data")
            filename = rel_path.replace("\\", "/")  # Normalize path separators
            additional_meta = train_metadata.get(filename, {})

            if existing_signal_files:
                # Load existing spectrograms
                for file in existing_signal_files:
                    spec = torch.load(file)
                    signal_specs.append(spec)

                    # Create metadata entry with both sources
                    meta_entry = {
                        "original_file": audio_file,
                        "processed_file": file,
                        "folder": folder,
                        "type": "signal",
                    }

                    # Add additional metadata from train.csv
                    for key, value in additional_meta.items():
                        if key not in meta_entry:  # Don't overwrite existing keys
                            meta_entry[f"train_{key}"] = value

                    metadata.append(meta_entry)

                # Load existing noise specs if available
                noise_pattern = str(noise_dir / f"{base_filename}_*.pt")
                for file in glob.glob(noise_pattern):
                    spec = torch.load(file)
                    noise_specs.append(spec)

                    # Create metadata entry with both sources
                    meta_entry = {
                        "original_file": audio_file,
                        "processed_file": file,
                        "folder": folder,
                        "type": "noise",
                    }

                    # Add additional metadata from train.csv
                    for key, value in additional_meta.items():
                        if key not in meta_entry:  # Don't overwrite existing keys
                            meta_entry[f"train_{key}"] = value

                    metadata.append(meta_entry)

                continue

            try:
                # Process audio file into chunks
                signal_chunks, noise_chunks = preprocessor.process_audio(
                    audio_file, visualization_dir="visualizations"
                )

                # Save signal chunks
                for i, chunk in enumerate(signal_chunks):
                    chunk_file = signal_dir / f"{base_filename}_{i:03d}.pt"
                    torch.save(chunk, chunk_file)
                    signal_specs.append(chunk)

                    # Create metadata entry with both sources
                    meta_entry = {
                        "original_file": audio_file,
                        "processed_file": str(chunk_file),
                        "folder": folder,
                        "type": "signal",
                        "chunk_index": i,
                    }

                    # Add additional metadata from train.csv
                    for key, value in additional_meta.items():
                        if key not in meta_entry:  # Don't overwrite existing keys
                            meta_entry[f"train_{key}"] = value

                    metadata.append(meta_entry)

                # Save noise chunks
                for i, chunk in enumerate(noise_chunks):
                    chunk_file = noise_dir / f"{base_filename}_{i:03d}.pt"
                    torch.save(chunk, chunk_file)
                    noise_specs.append(chunk)

                    # Create metadata entry with both sources
                    meta_entry = {
                        "original_file": audio_file,
                        "processed_file": str(chunk_file),
                        "folder": folder,
                        "type": "noise",
                        "chunk_index": i,
                    }

                    # Add additional metadata from train.csv
                    for key, value in additional_meta.items():
                        if key not in meta_entry:  # Don't overwrite existing keys
                            meta_entry[f"train_{key}"] = value

                    metadata.append(meta_entry)

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

    # Print metadata summary
    if "train_rating" in metadata_df.columns:
        print("\nMetadata Summary:")
        print(f"Files with rating: {metadata_df['train_rating'].notna().sum()}")
        print(f"Average rating: {metadata_df['train_rating'].mean():.2f}")

    if "train_collection" in metadata_df.columns:
        print("\nCollection Distribution:")
        print(metadata_df["train_collection"].value_counts().to_string())

    return signal_specs, noise_specs, metadata_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess bird audio files")
    parser.add_argument(
        "--input_dir",
        "-I",
        type=str,
        default="data/train_audio",
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--output_dir",
        "-O",
        type=str,
        default="train_audio_processed",
        help="Directory to save processed files",
    )
    parser.add_argument(
        "--metadata",
        "-M",
        type=str,
        default="data/train.csv",
        help="Path to train.csv with additional metadata",
    )

    args = parser.parse_args()

    # Get all .ogg files recursively
    audio_files = glob.glob(f"{args.input_dir}/**/*.ogg", recursive=True)
    print(f"Found {len(audio_files)} audio files")

    # Process files
    prepare_batch(audio_files, metadata_path=args.metadata, save_dir=args.output_dir)
