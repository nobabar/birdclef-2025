import glob
import os
from pathlib import Path
from queue import Queue
from threading import Lock

import numpy as np
import polars as pl
import torch
import torchaudio
import torchaudio.transforms as AT
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from scipy import ndimage
from silero_vad import get_speech_timestamps, load_silero_vad

from waveform_comparaison import (
    compare_energy,
    visualize_waveform_in_seconds,
)

console = Console()
error_queue = Queue()
error_lock = Lock()


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
            waveform with human voice segments removed (time shortened)
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

        # Invert speech segments to get non-voice segments (to keep)
        non_voice_segments = []
        prev_end = 0
        for segment in speech_timestamps:
            start = int(segment["start"] * self.sample_rate / 16000)
            end = int(segment["end"] * self.sample_rate / 16000)
            if prev_end < start:
                non_voice_segments.append((prev_end, start))
            prev_end = end
        if prev_end < waveform.shape[1]:
            non_voice_segments.append((prev_end, waveform.shape[1]))

        # Concatenate the kept parts
        kept_waveform = torch.cat(
            [waveform[:, start:end] for start, end in non_voice_segments], dim=1
        )

        return kept_waveform

    def process_audio(
        self,
        audio_path,
        chunk_duration=3.0,
        overlap=0.5,
        visualization_dir=None,
        debug_mode=False,
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
            visualization_dir: Directory to save visualizations
            debug_mode: Whether to run in debug mode

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

        # Only do visualization and comparison in debug mode
        if debug_mode:
            compare_energy(original_waveform, waveform)
            if visualization_dir:
                filename = Path(audio_path).stem
                visualization_path = os.path.join(
                    visualization_dir, f"{filename}_waveform.png"
                )
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


def process_single_file(args):
    """
    Process a single audio file. Helper function for parallel processing.

    Args:
        args: Tuple containing (audio_file, save_dir, folder, additional_meta)

    Returns:
        List of metadata entries for the processed file
    """
    audio_file, save_dir, folder, additional_meta = args
    metadata_entries = []

    try:
        # Create preprocessor for this process
        preprocessor = BirdSongPreprocessor()

        # Create unique filenames for the processed specs
        base_filename = Path(audio_file).stem
        signal_dir = save_dir / folder / "signal"
        noise_dir = save_dir / folder / "noise"

        signal_dir.mkdir(exist_ok=True, parents=True)
        noise_dir.mkdir(exist_ok=True, parents=True)

        # Process audio file into chunks
        signal_chunks, noise_chunks = preprocessor.process_audio(
            audio_file, debug_mode=False
        )

        # Save signal chunks
        for i, chunk in enumerate(signal_chunks):
            chunk_file = signal_dir / f"{base_filename}_{i:03d}.pt"
            torch.save(chunk, chunk_file)

            meta_entry = {
                "original_file": audio_file,
                "processed_file": str(chunk_file),
                "folder": folder,
                "type": "signal",
                "chunk_index": i,
            }

            for key, value in additional_meta.items():
                if key not in meta_entry:
                    meta_entry[f"train_{key}"] = value

            metadata_entries.append(meta_entry)

        # Save noise chunks
        for i, chunk in enumerate(noise_chunks):
            chunk_file = noise_dir / f"{base_filename}_{i:03d}.pt"
            torch.save(chunk, chunk_file)

            meta_entry = {
                "original_file": audio_file,
                "processed_file": str(chunk_file),
                "folder": folder,
                "type": "noise",
                "chunk_index": i,
            }

            for key, value in additional_meta.items():
                if key not in meta_entry:
                    meta_entry[f"train_{key}"] = value

            metadata_entries.append(meta_entry)

    except Exception as e:
        error_msg = f"Error processing {audio_file}: {str(e)}"
        error_queue.put(error_msg)

    return metadata_entries


def prepare_batch(
    audio_files,
    metadata_path="data/train.csv",
    save_dir="train_audio_processed",
    num_workers=None,  # None means use all available cores
):
    """
    Prepare a batch of audio files for model training or inference.

    Args:
        audio_files (list): List of audio file paths
        metadata_path (str): Path to the train.csv file with additional metadata
        save_dir (str): Directory to save the processed audio files
        num_workers (int): Number of parallel workers to use. None means use all available cores.
    """
    import multiprocessing as mp

    # Create save_dir if it doesn't exist
    save_dir = Path("data", save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize metadata_entries list
    metadata_entries = []

    # Group files by folder
    files_by_folder = {}
    for file in audio_files:
        folder = os.path.basename(os.path.dirname(file))
        if folder not in files_by_folder:
            files_by_folder[folder] = []
        files_by_folder[folder].append(file)

    # Load train.csv metadata if available
    if metadata_path and os.path.exists(metadata_path):
        try:
            # Load train.csv with explicit schema matching all columns
            train_metadata_df = pl.read_csv(
                metadata_path,
                schema={
                    "primary_label": pl.Utf8,
                    "secondary_labels": pl.Utf8,
                    "type": pl.Utf8,
                    "filename": pl.Utf8,
                    "collection": pl.Utf8,
                    "rating": pl.Float64,
                    "url": pl.Utf8,
                    "latitude": pl.Float64,
                    "longitude": pl.Float64,
                    "scientific_name": pl.Utf8,
                    "common_name": pl.Utf8,
                    "author": pl.Utf8,
                    "license": pl.Utf8,
                },
                null_values=["", "['']", "None"],
            )

            # Clean data using Polars expressions
            train_metadata_df = (
                train_metadata_df.filter(
                    pl.col("primary_label").is_not_null()
                    & (pl.col("primary_label") != "")
                )
                .with_columns(
                    [
                        pl.col("secondary_labels").fill_null("[]"),
                    ]
                )
                .select(["filename", "primary_label", "secondary_labels"])
            )

            # Convert to LazyFrame and cache the sorted result
            train_metadata_df = train_metadata_df.lazy().sort("filename").cache()

            console.print(
                f"\n[green]Loaded metadata for {train_metadata_df.select(pl.len()).collect().item()} files from train.csv[/green]"
            )

            # Process each folder to collect metadata from existing processed files
            for folder, folder_files in files_by_folder.items():
                signal_dir = save_dir / folder / "signal"
                noise_dir = save_dir / folder / "noise"

                # Add signal files metadata
                for audio_file in folder_files:
                    rel_path = os.path.relpath(audio_file, "data/train_audio_data")
                    filename = "/".join(rel_path.replace("\\", "/").split("/")[-2:])

                    meta = (
                        train_metadata_df.filter(pl.col("filename") == filename)
                        .select(["primary_label", "secondary_labels"])
                        .collect()
                    )

                    if meta.height > 0:
                        meta_row = meta.row(0, named=True)  # Get as dictionary
                    else:
                        meta_row = {"primary_label": "", "secondary_labels": "[]"}

                    signal_files = list(
                        signal_dir.glob(f"{Path(audio_file).stem}_*.pt")
                    )
                    for file in signal_files:
                        entry = {
                            "original_file": audio_file,
                            "processed_file": str(file),
                            "folder": folder,
                            "type": "signal",
                            "chunk_index": int(file.stem.split("_")[-1]),
                            "primary_label": meta_row["primary_label"],
                            "secondary_labels": meta_row["secondary_labels"],
                        }
                        metadata_entries.append(entry)

        except Exception as e:
            console.print(f"[red]Error loading train.csv metadata: {e}[/red]")
            raise

    # Prepare arguments for parallel processing
    process_args = []
    for folder, folder_files in files_by_folder.items():
        for audio_file in folder_files:
            # Check if already processed
            signal_dir = save_dir / folder / "signal"
            signal_pattern = str(signal_dir / f"{Path(audio_file).stem}_*.pt")
            if not glob.glob(signal_pattern):  # Only process if not already done
                rel_path = os.path.relpath(audio_file, "data/train_audio_data")
                filename = "/".join(rel_path.replace("\\", "/").split("/")[-2:])
                meta = train_metadata_df.filter(pl.col("filename") == filename)
                process_args.append((audio_file, save_dir, folder, meta))

    # Create overall progress tracking
    overall_progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
        transient=False,
    )

    # Create a function to get a new progress bar with consistent style
    def create_progress():
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True,
        )

    file_progress = create_progress()

    # Get terminal size and adjust error panel size
    terminal_height = console.size.height
    error_panel_size = min(10, terminal_height // 3)

    # Initialize error panel
    error_panel = Panel(
        "No errors yet",
        title="Recent Errors",
        border_style="red",
        padding=(0, 1),
    )

    # Create progress group
    progress_group = Group(overall_progress, file_progress, error_panel)

    total_steps = 3  # 1. Process files 2. Collect metadata 3. Save results
    overall_task = overall_progress.add_task("Overall progress", total=total_steps)

    metadata_entries = []  # List to collect all metadata entries

    with Live(
        progress_group,
        refresh_per_second=10,
        console=console,
        transient=False,
        auto_refresh=False,
    ) as live:
        # Step 1: Process new files if any
        if process_args:
            file_task = file_progress.add_task(
                "Processing new files", total=len(process_args)
            )
            errors = []

            with mp.Pool(num_workers) as pool:
                try:
                    for result in pool.imap_unordered(
                        process_single_file, process_args
                    ):
                        metadata_entries.extend(result)
                        file_progress.update(file_task, advance=1)
                        live.refresh()

                        # Check for new errors
                        while not error_queue.empty():
                            error = error_queue.get()
                            errors.append(error)
                            error_messages = []
                            for err in errors[-error_panel_size:]:
                                max_width = console.width - 10
                                if len(err) > max_width:
                                    err = err[: max_width - 3] + "..."
                                error_messages.append(err)

                            error_panel.renderable = (
                                "\n".join(error_messages)
                                if error_messages
                                else "No errors yet"
                            )
                            error_panel.title = f"Recent Errors ({len(errors)} total)"
                            live.refresh()
                finally:
                    file_progress.update(file_task, completed=len(process_args))
                    live.refresh()

        overall_progress.update(overall_task, advance=1)
        live.refresh()

        # Step 2: Collect metadata from existing files
        # Create new progress bar for metadata collection
        file_progress = create_progress()
        progress_group.renderables[1] = (
            file_progress  # Update the progress bar in the group
        )
        file_task = file_progress.add_task(
            "Collecting metadata from existing files", total=len(files_by_folder)
        )
        live.refresh()

        # Metadata collection
        for folder, folder_files in files_by_folder.items():
            signal_dir = save_dir / folder / "signal"
            noise_dir = save_dir / folder / "noise"

            # Process all files in the folder at once
            for audio_file in folder_files:
                # Get metadata for this file
                rel_path = os.path.relpath(audio_file, "data/train_audio_data")
                filename = "/".join(rel_path.replace("\\", "/").split("/")[-2:])

                meta = (
                    train_metadata_df.filter(pl.col("filename") == filename)
                    .select(["primary_label", "secondary_labels"])
                    .collect()
                )

                if meta.height > 0:
                    meta_row = meta.row(0, named=True)  # Get as dictionary
                else:
                    meta_row = {"primary_label": "", "secondary_labels": "[]"}

                # Get all processed files for this audio file
                signal_files = list(signal_dir.glob(f"{Path(audio_file).stem}_*.pt"))
                noise_files = list(noise_dir.glob(f"{Path(audio_file).stem}_*.pt"))

                # Add signal files metadata
                for file in signal_files:
                    entry = {
                        "original_file": audio_file,
                        "processed_file": str(file),
                        "folder": folder,
                        "type": "signal",
                        "chunk_index": int(file.stem.split("_")[-1]),
                        "primary_label": meta_row["primary_label"],
                        "secondary_labels": meta_row["secondary_labels"],
                    }
                    metadata_entries.append(entry)

                # Add noise files metadata with the same labels
                for file in noise_files:
                    entry = {
                        "original_file": audio_file,
                        "processed_file": str(file),
                        "folder": folder,
                        "type": "noise",
                        "chunk_index": int(file.stem.split("_")[-1]),
                        "primary_label": meta_row["primary_label"],
                        "secondary_labels": meta_row["secondary_labels"],
                    }
                    metadata_entries.append(entry)

                file_progress.update(file_task, advance=1)
                live.refresh()

            overall_progress.update(overall_task, advance=1)
            live.refresh()

        # Step 3: Save metadata
        # Create new progress bar for saving
        file_progress = create_progress()
        progress_group.renderables[1] = (
            file_progress  # Update the progress bar in the group
        )
        file_task = file_progress.add_task("Saving metadata to CSV", total=1)
        live.refresh()

        # Create DataFrame and save
        metadata_df = pl.DataFrame(metadata_entries)
        metadata_df.write_csv(save_dir / "metadata.csv")

        file_progress.update(file_task, completed=1)
        overall_progress.update(overall_task, advance=1)
        live.refresh()

        console.print("\n[bold blue]Processing Summary:[/bold blue]")
        console.print(f"Total files processed: {len(metadata_df)}")
        folder_counts = (
            metadata_df.filter(pl.col("type") == "signal")
            .group_by("folder")
            .len()
            .sort("len", descending=True)
        )
        console.print(folder_counts.head())

    return metadata_df


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
