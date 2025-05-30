import os

import matplotlib.pyplot as plt
import torch


def compare_energy(original, cleaned):
    """
    Compare and display the total energy of the original and cleaned waveforms.

    The energy is calculated as the sum of squared amplitudes.
    Useful for understanding how much signal (e.g., speech) was removed.

    Args:
    ----
        original (torch.Tensor): Original waveform of shape [channels, num_samples].
        cleaned (torch.Tensor): Cleaned waveform of shape [channels, num_samples].
    """
    original_energy = torch.sum(original**2).item()
    cleaned_energy = torch.sum(cleaned**2).item()
    print(f"\n[DEBUG] Energy before: {original_energy:.2f}")
    print(f"[DEBUG] Energy after : {cleaned_energy:.2f}")
    print(f"[DEBUG] Energy removed: {original_energy - cleaned_energy:.2f}")


def visualize_waveform(original, cleaned):
    """
    Plot the waveform of original and cleaned audio signals.

    When loading an audio file with torchaudio, the waveform is a tensor of shape:
        [channels, num_samples]
    - channels: 1 for mono, 2 for stereo
    - num_samples: total number of audio samples

    The x-axis shows the sample index, which represents the position in the audio signal.

    Args:
    ----
        original (torch.Tensor): Original waveform of shape [channels, num_samples].
        cleaned (torch.Tensor): Cleaned waveform of shape [channels, num_samples].
    """
    plt.figure(figsize=(15, 4))
    plt.plot(original[0].cpu().numpy(), label="Original", alpha=0.6)
    plt.plot(cleaned[0].cpu().numpy(), label="After Voice Removal", alpha=0.6)
    plt.legend()
    plt.title("Waveform Before and After Voice Removal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def visualize_waveform_in_seconds(original, cleaned, sample_rate, save_path=None):
    """
    Visualize the waveform of audio signals before and after processing, with time represented in seconds.

    Args:
    ----
        original (torch.Tensor): Original audio waveform (1, N).
        cleaned (torch.Tensor): Processed audio waveform (1, M).
        sample_rate (int): Sample rate in Hz.
        save_path (str or None): If provided, save the plot to this path.

    Returns:
    -------
        None
    """
    # Prepare time axes
    original_len = original.shape[1]
    cleaned_len = cleaned.shape[1]

    time_original = torch.linspace(0, original_len / sample_rate, steps=original_len)
    time_cleaned = torch.linspace(0, cleaned_len / sample_rate, steps=cleaned_len)

    plt.figure(figsize=(15, 4))
    plt.plot(time_original, original[0].cpu().numpy(), label="Original", alpha=0.6)
    plt.plot(
        time_cleaned, cleaned[0].cpu().numpy(), label="After Voice Removal", alpha=0.6
    )
    plt.legend()
    plt.title("Waveform Before and After Voice Removal")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def normalize_waveform(waveform):
    """
    Normalize the waveform to have values between -1 and 1.

    This is useful for ensuring that the waveform is within a standard range,
    especially before saving or processing.

    Args:
    ----
        waveform (torch.Tensor): The input waveform tensor of shape [channels, num_samples].

    Returns:
    -------
        torch.Tensor: The normalized waveform tensor of the same shape.
    """
    max_val = waveform.abs().max()
    return waveform / (max_val + 1e-9) if max_val > 0 else waveform
