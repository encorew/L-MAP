import numpy as np
from numpy.fft import fft, ifft
from scipy.signal import find_peaks, peak_widths, chirp

def multi_fft(seq):
    real_freq_seq = []
    normal_freq_seq = []
    for i in range(seq.shape[0]):
        freq = abs(fft(seq[i, :]))
        freq = freq[1:len(freq) // 2]
        real_freq_seq.append(freq)
        # normal_freq_seq.append(fft(seq[:, i]))
    return np.array(real_freq_seq).T


def multi_ifft(frequency):
    # shape为（time_length,dimension)
    reconstructed_series = []
    for i in range(frequency.shape[1]):
        series = ifft(frequency[:, i])
        reconstructed_series.append(series)
    return np.array(reconstructed_series).T


def reconstruct_time_series(input_time_series):
    N = len(input_time_series)
    new_time_series = []
    for i in range(input_time_series.shape[1]):
        fft_result = np.fft.fft(input_time_series[:, i])
        frequencies = np.fft.fftfreq(N)
        amplitude = np.abs(fft_result)
        peaks, _ = find_peaks(amplitude, height=0.1)
        peak_freqs = frequencies[peaks]
        widths, _, _, _ = peak_widths(amplitude, peaks)
        selected_peak = np.random.choice(len(peaks))
        time_series = chirp(np.arange(N), f0=0, f1=peak_freqs[selected_peak], t1=N, method='quadratic')
        amplitude_ratio = np.max(input_time_series) / np.max(time_series)
        time_series *= amplitude_ratio
        new_time_series.append(time_series[:N])
    return np.array(new_time_series).T


def freq_distance(seq1, seq2):
    return np.sum((seq1 - seq2) ** 2)


def fft_reconstruction(seq):
    fft = multi_fft(seq)
