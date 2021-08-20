import cupy.linalg
import numpy as np
import pywt
import scipy.linalg, scipy.signal


def get_spectrum(signal, fs, freq_range=60):
    window_size = round(fs / 8 + 1)
    if not window_size % 2:
        window_size = window_size + 1
    # window0 = np.exp(np.log(0.005) * np.linspace(-1, 1, window_size) ** 2)

    window = scipy.signal.get_window("hamming", window_size, fftbins=True)
    freq_time_prof_allfreq, _ = tfrsp(signal, fs, window)

    freq_bins_ = np.concatenate([np.arange(0, fs / 2), np.arange(-fs / 2, 0)]) / fs

    freq_lowpassfreq_selected = np.logical_and(freq_bins_ <= freq_range / fs, freq_bins_ >= -freq_range / fs)
    spectrum = freq_time_prof_allfreq[freq_lowpassfreq_selected, :]
    spectrum = abs(spectrum) / abs(spectrum).mean(0)
    spectrum = np.fft.fftshift(spectrum, axes=0)
    return spectrum


def tfrsp(signal, n_fbins, fwindow):
    timestamps = np.arange(len(signal), dtype=int) + 1
    if n_fbins % 2 == 0:
        freqs = np.hstack((np.arange(n_fbins / 2), np.arange(-n_fbins / 2, 0)))
    else:
        freqs = np.hstack((np.arange((n_fbins - 1) / 2), np.arange(-(n_fbins - 1) / 2, 0)))
    freqs = freqs / n_fbins
    tfr = np.zeros((n_fbins, timestamps.shape[0]), dtype=complex)
    # ===============================================
    lh = (fwindow.shape[0] - 1) // 2
    rangemin = min([round(n_fbins / 2.0) - 1, lh])
    starts = -np.min(np.c_[rangemin * np.ones(timestamps.shape), timestamps - 1], axis=1).astype(int)
    ends = np.min(np.c_[rangemin * np.ones(timestamps.shape), signal.shape[0] - timestamps], axis=1).astype(int)
    conj_fwindow = np.conj(fwindow)
    for icol in range(tfr.shape[1]):
        ti = timestamps[icol]
        start = starts[icol]
        end = ends[icol]
        tau = np.arange(start, end + 1, dtype=int)
        indices = np.remainder(n_fbins + tau, n_fbins)
        tfr[indices, icol] = signal[ti + tau - 1] * conj_fwindow[lh + tau] / np.linalg.norm(fwindow[lh + tau])
    tfr = np.abs(np.fft.fft(tfr, axis=0)) ** 2
    return tfr, freqs


def freq2scale(freq, w_name, fs):
    return pywt.central_frequency(w_name) / (freq * fs)


def complex_sign(complex):
    return complex / abs(complex)


def pca_matlab(X):

    center = X - X.mean(axis=0)
    # U, S, V = scipy.linalg.svd(center, full_matrices=False)
    U, S, V = np.linalg.svd(center, full_matrices=False)
    # score = U * S
    latent = S ** 2 / (X.shape[0] - 1)

    V = V.conj().T  # 使所有中间结果与matlab结果对齐

    # flip eigenvectors' sign to enforce deterministic output
    max_abs_idx = np.argmax(abs(V), axis=0)
    colsign = complex_sign(
        V[
            max_abs_idx,
            range(V.shape[0]),
        ]
    )
    U *= colsign
    V *= colsign[None, :]
    score = U * S
    # result:  center == U @ np.diag(S) @ V.conj().T
    coeff = V
    return coeff, score, latent


from .conj_multi import conj_multi_csi
from .signal_filter import gen_fir_filter, gen_iir_filter
