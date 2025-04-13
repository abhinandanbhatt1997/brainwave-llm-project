import mne
import numpy as np
from config.settings import FILTER_RANGE, SAMPLE_RATE

def load_raw_eeg(file_path: str):
    """Load .edf or .fif files"""
    return mne.io.read_raw_edf(file_path, preload=True)

def preprocess_eeg(raw):
    """Filter and epoch raw EEG"""
    raw.filter(*FILTER_RANGE, method='iir')
    return mne.make_fixed_length_epochs(raw, duration=2.0, overlap=1.0)

def extract_features(epochs):
    """Calculate alpha/beta power ratios"""
    psds, freqs = mne.time_frequency.psd_array_multitaper(
        epochs.get_data(), sfreq=SAMPLE_RATE, fmin=1, fmax=40
    )
    alpha = np.mean(psds[:, :, (freqs >= 8) & (freqs <= 12)], axis=(1, 2))
    beta = np.mean(psds[:, :, (freqs >= 13) & (freqs <= 30)], axis=(1, 2))
    return np.column_stack([alpha, beta])