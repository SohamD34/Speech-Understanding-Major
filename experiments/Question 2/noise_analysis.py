import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import librosa
import librosa.display
import pandas as pd

def load_audio(file_path, sr=None):
    """
    Load an audio file using librosa.
    
    Args:
        file_path: Path to the audio file
        sr: Target sampling rate (None for original)
        
    Returns:
        audio: Audio time series
        sr: Sampling rate
    """
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def calculate_snr(clean_audio, noisy_audio):
    """
    Calculate Signal-to-Noise Ratio (SNR).
    
    Args:
        clean_audio: Clean audio signal
        noisy_audio: Noisy audio signal
        
    Returns:
        snr: Signal-to-Noise Ratio in dB
    """
    if len(clean_audio) != len(noisy_audio):
        # Truncate to the same length
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]
        
    # Calculate noise
    noise = noisy_audio - clean_audio
    
    # Calculate signal and noise power
    signal_power = np.sum(clean_audio ** 2)
    noise_power = np.sum(noise ** 2)
    
    # Avoid division by zero
    if noise_power == 0:
        return float('inf')
    
    # Calculate SNR in dB
    snr = 10 * np.log10(signal_power / noise_power)
    
    return snr

def estimate_snr(audio):
    """
    Estimate SNR without clean reference by assuming quietest segments represent noise floor.
    
    Args:
        audio: Audio signal
        
    Returns:
        estimated_snr: Estimated Signal-to-Noise Ratio in dB
    """
    # Split audio into frames
    frame_length = 2048
    hop_length = 512
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    
    # Calculate energy of each frame
    frame_energy = np.sum(frames**2, axis=0)
    
    # Assume the 10% lowest energy frames represent noise
    noise_threshold = np.percentile(frame_energy, 10)
    noise_frames = frames[:, frame_energy <= noise_threshold]
    
    if noise_frames.size == 0:
        return None
    
    # Estimate noise power
    noise_power = np.mean(np.sum(noise_frames**2, axis=0))
    
    # Estimate signal power (exclude noise)
    signal_frames = frames[:, frame_energy > noise_threshold]
    
    if signal_frames.size == 0:
        return None
        
    signal_power = np.mean(np.sum(signal_frames**2, axis=0))
    
    # Calculate estimated SNR
    if noise_power == 0:
        return float('inf')
        
    estimated_snr = 10 * np.log10(signal_power / noise_power)
    
    return estimated_snr

def analyze_frequency_spectrum(audio, sr):
    """
    Analyze the frequency spectrum of an audio signal.
    
    Args:
        audio: Audio signal
        sr: Sampling rate
        
    Returns:
        dict: Dictionary containing frequency characteristics
    """
    # Compute the spectrogram
    S = np.abs(librosa.stft(audio))
    
    # Convert to dB scale
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    
    # Compute frequency bands (low, mid, high)
    freqs = librosa.fft_frequencies(sr=sr)
    low_freq_mask = freqs < 500
    mid_freq_mask = (freqs >= 500) & (freqs < 2000)
    high_freq_mask = freqs >= 2000
    
    # Calculate average energy in each frequency band
    low_energy = np.mean(np.mean(S[low_freq_mask, :]))
    mid_energy = np.mean(np.mean(S[mid_freq_mask, :]))
    high_energy = np.mean(np.mean(S[high_freq_mask, :]))
    
    # Calculate spectral centroid (brightness)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0].mean()
    
    # Calculate spectral bandwidth (spread)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0].mean()
    
    # Calculate spectral flatness (noisiness)
    spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0].mean()
    
    return {
        'low_freq_energy': low_energy,
        'mid_freq_energy': mid_energy,
        'high_freq_energy': high_energy,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_flatness': spectral_flatness
    }

def analyze_paired_data(clean_dir, noisy_dir):
    """
    Analyze paired clean and noisy audio files to compute noise metrics.
    
    Args:
        clean_dir: Directory containing clean audio files
        noisy_dir: Directory containing corresponding noisy audio files
        
    Returns:
        pd.DataFrame: DataFrame with noise analysis results
    """
    results = []
    
    # Get list of clean files
    clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith(('.wav', '.mp3', '.flac'))])
    
    for clean_file in clean_files:
        file_base = os.path.splitext(clean_file)[0]
        
        # Find corresponding noisy file (assuming same filename)
        noisy_file = None
        for ext in ['.wav', '.mp3', '.flac']:
            if os.path.exists(os.path.join(noisy_dir, file_base + ext)):
                noisy_file = file_base + ext
                break
        
        if not noisy_file:
            print(f"No matching noisy file found for: {clean_file}")
            continue
            
        # Load audio files
        clean_audio, sr_clean = load_audio(os.path.join(clean_dir, clean_file))
        noisy_audio, sr_noisy = load_audio(os.path.join(noisy_dir, noisy_file))
        
        if clean_audio is None or noisy_audio is None:
            continue
            
        # Ensure same sampling rate
        if sr_clean != sr_noisy:
            noisy_audio = librosa.resample(noisy_audio, orig_sr=sr_noisy, target_sr=sr_clean)
            sr_noisy = sr_clean
            
        # Calculate SNR
        snr = calculate_snr(clean_audio, noisy_audio)
        
        # Analyze frequency spectrum of noise
        noise = noisy_audio - clean_audio  # Extract noise
        noise_freq_analysis = analyze_frequency_spectrum(noise, sr_clean)
        
        # Analyze original and noisy audio signals
        clean_freq_analysis = analyze_frequency_spectrum(clean_audio, sr_clean)
        noisy_freq_analysis = analyze_frequency_spectrum(noisy_audio, sr_noisy)
        
        # Store results
        result = {
            'file_name': clean_file,
            'snr_db': snr,
            'clean_spectral_flatness': clean_freq_analysis['spectral_flatness'],
            'noisy_spectral_flatness': noisy_freq_analysis['spectral_flatness'],
            'noise_low_energy': noise_freq_analysis['low_freq_energy'],
            'noise_mid_energy': noise_freq_analysis['mid_freq_energy'],
            'noise_high_energy': noise_freq_analysis['high_freq_energy'],
            'noise_spectral_centroid': noise_freq_analysis['spectral_centroid']
        }
        
        results.append(result)
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def analyze_noisy_only(noisy_dir):
    """
    Analyze noisy audio files without clean references.
    
    Args:
        noisy_dir: Directory containing noisy audio files
        
    Returns:
        pd.DataFrame: DataFrame with noise analysis results
    """
    results = []
    
    # Get list of noisy files
    noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith(('.wav', '.mp3', '.flac'))])
    
    for noisy_file in noisy_files:
        # Load audio file
        noisy_audio, sr = load_audio(os.path.join(noisy_dir, noisy_file))
        
        if noisy_audio is None:
            continue
            
        # Estimate SNR
        estimated_snr = estimate_snr(noisy_audio)
        
        # Analyze frequency spectrum
        freq_analysis = analyze_frequency_spectrum(noisy_audio, sr)
        
        # Compute additional noise metrics
        rms = np.sqrt(np.mean(noisy_audio**2))
        peak = np.max(np.abs(noisy_audio))
        crest_factor = peak / rms if rms > 0 else float('inf')
        
        # Store results
        result = {
            'file_name': noisy_file,
            'estimated_snr_db': estimated_snr,
            'rms_amplitude': rms,
            'peak_amplitude': peak,
            'crest_factor': crest_factor,
            'spectral_flatness': freq_analysis['spectral_flatness'],
            'spectral_centroid': freq_analysis['spectral_centroid'],
            'spectral_bandwidth': freq_analysis['spectral_bandwidth'],
            'low_freq_energy': freq_analysis['low_freq_energy'],
            'mid_freq_energy': freq_analysis['mid_freq_energy'],
            'high_freq_energy': freq_analysis['high_freq_energy']
        }
        
        results.append(result)
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def visualize_noise_spectrum(audio_path, title=None, save_path=None):
    """
    Visualize the frequency spectrum of an audio file.
    
    Args:
        audio_path: Path to the audio file
        title: Title for the plot
        save_path: Path to save the plot (optional)
    """
    # Load audio
    audio, sr = load_audio(audio_path)
    
    if audio is None:
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr)
    plt.title('Waveform')
    
    # Plot spectrogram
    plt.subplot(2, 1, 2)
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def generate_report(paired_results=None, noisy_only_results=None, output_path='noise_analysis_report.csv'):
    """
    Generate a comprehensive report from the analysis results.
    
    Args:
        paired_results: Results from paired analysis
        noisy_only_results: Results from noisy-only analysis
        output_path: Path to save the report
    """
    # Combine results if both are available
    if paired_results is not None and noisy_only_results is not None:
        # Create a unified report with a column indicating the source
        paired_results['source'] = 'paired'
        noisy_only_results['source'] = 'noisy_only'
        
        # Ensure columns match
        common_columns = set(paired_results.columns).intersection(set(noisy_only_results.columns))
        all_columns = list(common_columns) + ['source']
        
        # Combine dataframes
        combined_results = pd.concat([
            paired_results[all_columns], 
            noisy_only_results[all_columns]
        ])
        
        # Save report
        combined_results.to_csv(output_path, index=False)
        print(f"Combined report saved to {output_path}")
        
        return combined_results
    
    # If only one type of analysis was performed
    elif paired_results is not None:
        paired_results.to_csv(output_path, index=False)
        print(f"Paired analysis report saved to {output_path}")
        return paired_results
    
    elif noisy_only_results is not None:
        noisy_only_results.to_csv(output_path, index=False)
        print(f"Noisy-only analysis report saved to {output_path}")
        return noisy_only_results
    
    else:
        print("No results to report")
        return None

def main():
    """
    Main function to run the noise analysis on the provided directory structure.
    """
    data_dir = 'data'
    set1_dir = os.path.join(data_dir, 'denoising', 'set 1 - Clean and noisy')
    clean_dir = os.path.join(set1_dir, 'clean')
    noisy_dir1 = os.path.join(set1_dir, 'noisy')
    
    set2_dir = os.path.join(data_dir, 'denoising', 'set 2 - only noisy')
    
    for dir_path in [clean_dir, noisy_dir1, set2_dir]:
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
    
    paired_results = None
    if os.path.exists(clean_dir) and os.path.exists(noisy_dir1):
        print("Analyzing paired clean and noisy audio...")
        paired_results = analyze_paired_data(clean_dir, noisy_dir1)
        print(f"Paired analysis complete: {len(paired_results)} files processed")
    
    noisy_only_results = None
    if os.path.exists(set2_dir):
        print("Analyzing noisy-only audio...")
        noisy_only_results = analyze_noisy_only(set2_dir)
        print(f"Noisy-only analysis complete: {len(noisy_only_results)} files processed")
    
    generate_report(paired_results, noisy_only_results)
    
    print("Generating visualizations for sample files...")
    
    if os.path.exists(noisy_dir1):
        noisy_files = [f for f in os.listdir(noisy_dir1) if f.endswith(('.wav', '.mp3', '.flac'))]
        if noisy_files:
            sample_file = os.path.join(noisy_dir1, noisy_files[0])
            visualize_noise_spectrum(sample_file, 
                                    title=f"Noise Analysis - {noisy_files[0]}",
                                    save_path="results/sample_noise_spectrum_set1.png")
    
    if os.path.exists(set2_dir):
        noisy_files = [f for f in os.listdir(set2_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
        if noisy_files:
            sample_file = os.path.join(set2_dir, noisy_files[0])
            visualize_noise_spectrum(sample_file, 
                                    title=f"Noise Analysis - {noisy_files[0]}",
                                    save_path="results/sample_noise_spectrum_set2.png")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()