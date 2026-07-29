import numpy as np
from scipy import signal
from scipy.signal import find_peaks

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Create a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply the bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # Using filtfilt ensures zero phase distortion
    y = signal.filtfilt(b, a, data)
    return y

def extract_envelope(filtered_audio, fs, lowpass_cutoff=5.0):
    """
    Extract the envelope of the signal using Shannon Energy and a lowpass filter.
    Useful for making peaks (heartbeats/breaths) stand out.
    """
    # 1. Normalize
    normalized = filtered_audio / (np.max(np.abs(filtered_audio)) + 1e-6)
    
    # 2. Shannon Energy (emphasizes higher intensity signals)
    energy = (normalized ** 2) * np.log(normalized ** 2 + 1e-6)
    
    # 3. Smooth with a lowpass filter
    nyq = 0.5 * fs
    b, a = signal.butter(2, lowpass_cutoff / nyq, btype='low')
    envelope = signal.filtfilt(b, a, energy)
    
    # Absolute value just in case, though energy is positive
    return np.abs(envelope)

def calculate_rates_dsp(audio_data, fs):
    """
    Calculates Heart Rate (HR) and Respiratory Rate (RR) using purely DSP.
    Returns: HR (bpm), RR (bpm)
    """
    # 1. Filter the signals
    # Heart sounds (PCG): 20 Hz to 150 Hz
    pcg_signal = apply_bandpass_filter(audio_data, lowcut=20.0, highcut=150.0, fs=fs)
    
    # Lung sounds (Respiratory): 150 Hz to 800 Hz
    lung_signal = apply_bandpass_filter(audio_data, lowcut=150.0, highcut=800.0, fs=fs)
    
    # 2. Extract Envelopes
    # Heart rate is typically 1-3 Hz (60-180 BPM), so a 5Hz lowpass on the envelope is fine
    pcg_env = extract_envelope(pcg_signal, fs, lowpass_cutoff=5.0)
    
    # Breathing is roughly 0.2 - 0.5 Hz (12-30 BPM), so a very low 1Hz lowpass
    lung_env = extract_envelope(lung_signal, fs, lowpass_cutoff=1.5)
    
    # 3. Peak Detection for Heart Rate
    # Distance: at least 0.3 seconds between beats (max 200 BPM)
    min_distance_hr = int(0.3 * fs) 
    hr_peaks, _ = find_peaks(pcg_env, distance=min_distance_hr, prominence=np.std(pcg_env))
    
    hr_bpm = 0
    if len(hr_peaks) > 1:
        # Calculate diffs in seconds, then convert to BPM
        hr_intervals = np.diff(hr_peaks) / fs
        hr_bpm = 60.0 / np.mean(hr_intervals)
        
    # 4. Peak Detection for Respiratory Rate
    # Distance: at least 1.5 seconds between breaths (max 40 BPM)
    min_distance_rr = int(1.5 * fs)
    rr_peaks, _ = find_peaks(lung_env, distance=min_distance_rr, prominence=np.std(lung_env))
    
    rr_bpm = 0
    if len(rr_peaks) > 1:
        rr_intervals = np.diff(rr_peaks) / fs
        rr_bpm = 60.0 / np.mean(rr_intervals)
        
    return hr_bpm, rr_bpm

