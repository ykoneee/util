from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy import signal, io

from widar_reinpl_series.re_inpl_widar3 import pca_matlab, tfrsp


def dfs(csi_data, samp_rate=1000):
    rx_acnt = 3  # Antenna count for each receiver
    method = "stft"
    # method = "cwt"
    samp_rate -= samp_rate % 2
    # samp_rate = 1000
    half_rate = samp_rate / 2
    uppe_orde = 6
    uppe_stop = 45
    lowe_orde = 3
    lowe_stop = 2

    f = np.array(
        [
            0,
            uppe_stop / half_rate,
            uppe_stop * 1.1 / half_rate,
            1,
        ]
    )
    # f = np.array([0, 0.12, 0.3, 1])
    m = np.array([1, 1, 0, 0])
    b2 = signal.firls(49, f, m)

    [lu, ld] = signal.butter(uppe_orde, uppe_stop / half_rate, "lowpass")
    [hu, hd] = signal.butter(lowe_orde, lowe_stop / half_rate, "highpass")

    freq_bins_unwrap = (
        np.concatenate([np.arange(0, samp_rate / 2), np.arange(-samp_rate / 2, 0)])
        / samp_rate
    )

    freq_low_prof_select = np.logical_and(
        freq_bins_unwrap <= uppe_stop / samp_rate,
        freq_bins_unwrap >= -uppe_stop / samp_rate,
    )

    freq_lpf_positive_max = freq_low_prof_select[
        1 : len(freq_low_prof_select) // 2
    ].sum()
    freq_lpf_negative_min = freq_low_prof_select[len(freq_low_prof_select) // 2 :].sum()
    #

    # % Select Antenna Pair[WiDance]
    csi_mean = abs(csi_data).mean(0)
    csi_var = np.sqrt(abs(csi_data).var(0, ddof=1))
    csi_mean_var_ratio = csi_mean / csi_var
    idx = np.argmax(csi_mean_var_ratio.reshape(rx_acnt, 30).mean(1))
    # idx = 1
    csi_data_ref = np.tile(csi_data[:, idx * 30 : (idx + 1) * 30], (1, rx_acnt))

    # % Amp Adjust[IndoTrack]
    csi_data_adj = np.zeros_like(csi_data)
    csi_data_ref_adj = np.zeros_like(csi_data_ref)
    alpha_sum = 0
    for jj in range(30 * rx_acnt):
        amp = abs(csi_data[:, jj])
        alpha = np.min(amp[amp != 0])
        alpha_sum += alpha
        csi_data_adj[:, jj] = abs(amp - alpha) * np.exp(1j * np.angle(csi_data[:, jj]))

    beta = 1000 * alpha_sum / (30 * rx_acnt)
    for jj in range(30 * rx_acnt):
        csi_data_ref_adj[:, jj] = (abs(csi_data_ref[:, jj]) + beta) * np.exp(
            1j * np.angle(csi_data_ref[:, jj])
        )

    # Conj Mult==========================================================
    conj_mult = csi_data_adj * np.conj(csi_data_ref_adj)
    conj_mult = conj_mult[:, np.r_[: 30 * idx, 30 * (idx + 1) : rx_acnt * 30]]

    # Filter Out Static Component & High Frequency Component==========================================================
    # conj_mult -= conj_mult.mean(0)
    # conj_mult = signal.lfilter(hu, hd, conj_mult, axis=0)
    # conj_mult = signal.lfilter(lu, ld, conj_mult, axis=0)
    # conj_mult = signal.lfilter(b2, 1, conj_mult, axis=0)
    conj_mult = signal.filtfilt(b2, 1, conj_mult, axis=0, method="gust")
    # PCA analysis.======================================================================
    coeff, score, latent = pca_matlab(conj_mult)
    # conj_mult_pca = (conj_mult - conj_mult.mean(0)) @ coeff[:, 0]
    # conj_mult_pca = (conj_mult) @ coeff[:, 0]
    conj_mult_pca = score[:, 0]
    # plt.plot(conj_mult_pca)
    #  TFA ==============================================================================
    time_instance = np.arange(len(conj_mult_pca)) + 1
    window_size = round(samp_rate / 4 + 1)
    if not window_size % 2:
        window_size = window_size + 1
    window = np.exp(np.log(0.005) * np.linspace(-1, 1, window_size) ** 2)
    freq_time_prof_allfreq, _ = tfrsp(conj_mult_pca, time_instance, samp_rate, window)

    # Select Concerned Freq==========================================================================
    freq_time_prof: np.ndarray = freq_time_prof_allfreq[freq_low_prof_select, :]
    # Spectrum Normalization By Sum For Each Snapshot
    freq_time_prof = abs(freq_time_prof) / abs(freq_time_prof).sum(0)
    #
    # % Frequency Bin(Corresponding to FFT Results)
    freq_bin = np.r_[0 : freq_lpf_positive_max + 1, -1 * freq_lpf_negative_min : 0]

    doppler_spectrum = np.zeros(
        (
            1 + freq_lpf_positive_max + freq_lpf_negative_min,
            csi_data.shape[0],
        )
    )

    if freq_time_prof.shape[1] >= doppler_spectrum.shape[1]:
        doppler_spectrum = freq_time_prof[:, : doppler_spectrum.shape[1]]
    else:
        raise ValueError

    doppler_spectrum[:] = np.fft.fftshift(doppler_spectrum, axes=0)
    return doppler_spectrum


def widar3_dfs(imax):
    csi_path = Path(
        "/media/yk/linux_data/csi_dataset_survey/dataset/widar/mat_save_folder"
    )
    ret = []
    for i, n0 in enumerate(sorted(csi_path.glob("*.mat"))):
        print(i)
        csi_data = io.loadmat(n0)["csi_data"]
        ret.append(dfs(csi_data.reshape(-1, 90)))
        if i > imax:
            break
    return ret


if __name__ == "__main__":

    d = widar3_dfs(9)

    #     dp, pos = jl_ex_plcr(dd)
    #     res = process_f1(dp, pos)
    # plt.plot(res[:, 1], res[:, 0])

    fig = go.Figure()
    fig.add_traces([go.Heatmap(z=x, visible=False) for x in d])
    # for x in d:
    #     # fig.add_trace(go.Heatmap(z=x, visible=False))
    #
    #     fig.add_trace(go.Contour(z=x, visible=False))
    fig.data[0].visible = True

    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
            ],
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)

    sliders = [dict(active=0, pad={"t": len(steps)}, steps=steps)]

    fig.update_layout(sliders=sliders)

    fig.show()
    # fig.write_html("dfs.html", config={"displayModeBar": True})
