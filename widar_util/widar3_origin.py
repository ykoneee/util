from pathlib import Path

import numpy as np

# import seaborn as sb
import pywt
import scipy.io
import scipy.linalg
import scipy.signal

from util.widar_util import pca_matlab, tfrsp


def elesiver_np_process():
    root_path = Path("/media/yk/linux_data/csi_dataset_survey/dataset/Wih2h_dataset")
    volunteer_l = np.load(root_path / "volunteer_name_list.npy")

    for i, volunteer in enumerate(volunteer_l):
        ret = []
        print(volunteer)
        label = np.load(root_path / (volunteer + "_label.npy"), allow_pickle=True) - 1
        raw = np.load(root_path / (volunteer + "_raw.npy"), allow_pickle=True)[label == 0]
        for data in raw:
            csi_data = data["CSI"][:, 0, ...].reshape(-1, 90)
            # csi_data = data["CSI"].mean(1).reshape(-1, 90)

            sample_f = 1 / ((data["timestamp_low"][-1] - data["timestamp_low"][0]) * 1e-6 / (data["timestamp_low"].shape[0]))
            # print(sample_f)
            doppler_spectrum = dfs(csi_data, samp_rate=int(sample_f))
            # doppler_spectrum = dfs(csi_data,)
            ret.append(doppler_spectrum)
        break
    return ret


def dfs(csi_data, samp_rate=1000):
    rx_acnt = 3  # Antenna count for each receiver
    method = "stft"
    # method = "cwt"
    samp_rate -= samp_rate % 2
    # samp_rate = 1000
    half_rate = samp_rate / 2
    uppe_orde = 6
    uppe_stop = 60
    lowe_orde = 3
    lowe_stop = 2

    [lu, ld] = scipy.signal.butter(uppe_orde, uppe_stop / half_rate, "lowpass")
    [hu, hd] = scipy.signal.butter(lowe_orde, lowe_stop / half_rate, "highpass")

    freq_bins_unwrap = np.concatenate([np.arange(0, samp_rate / 2), np.arange(-samp_rate / 2, 0)]) / samp_rate

    freq_lpf_sele = np.logical_and(
        freq_bins_unwrap <= uppe_stop / samp_rate,
        freq_bins_unwrap >= -uppe_stop / samp_rate,
    )

    freq_lpf_positive_max = freq_lpf_sele[1 : len(freq_lpf_sele) // 2].sum()
    freq_lpf_negative_min = freq_lpf_sele[len(freq_lpf_sele) // 2 :].sum()
    #
    doppler_spectrum = np.zeros(
        (
            1 + freq_lpf_positive_max + freq_lpf_negative_min,
            csi_data.shape[0],
        )
    )

    # % Camera-Ready: Down Sample
    # csi_data = csi_data(round(1:1:size(csi_data,1)),:);

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
        csi_data_ref_adj[:, jj] = (abs(csi_data_ref[:, jj]) + beta) * np.exp(1j * np.angle(csi_data_ref[:, jj]))

    # % Conj Mult
    conj_mult = csi_data_adj * np.conj(csi_data_ref_adj)
    conj_mult = conj_mult[:, np.r_[: 30 * idx, 30 * (idx + 1) : rx_acnt * 30]]

    # % Filter Out Static Component & High Frequency Component

    conj_mult = scipy.signal.lfilter(hu, hd, conj_mult, axis=0)
    conj_mult = scipy.signal.lfilter(lu, ld, conj_mult, axis=0)

    # % PCA analysis.
    coeff, score, latent = pca_matlab(conj_mult)
    # conj_mult_pca = (conj_mult - conj_mult.mean(0)) @ coeff[:, 0]
    # conj_mult_pca = (conj_mult) @ coeff[:, 0]
    conj_mult_pca = score[:, 0]

    #  TFA With CWT or STFT
    if method == "cwt":
        assert False  # NEED FIX
        freq_time_prof_allfreq = pywt.cwt(
            conj_mult_pca,
            freq2scale(
                np.concatenate([np.arange(0, samp_rate / 2) + 1, np.arange(-samp_rate / 2, 0)]),
                "cmor4-1",
                1 / samp_rate,
            ),
            "cmor4-1",
        )
        # freq_time_prof_allfreq = scaled_cwt(conj_mult_pca,frq2scal([1:samp_rate/2 -1*samp_rate/2:-1],'cmor4-1',1/samp_rate),'cmor4-1');
    elif method == "stft":
        time_instance = np.arange(len(conj_mult_pca)) + 1
        window_size = round(samp_rate / 4 + 1)
        if not window_size % 2:
            window_size = window_size + 1
        window = np.exp(np.log(0.005) * np.linspace(-1, 1, window_size) ** 2)
        freq_time_prof_allfreq, _ = tfrsp(conj_mult_pca, time_instance, samp_rate, window)
    # % Select Concerned Freq
    freq_time_prof: np.ndarray = freq_time_prof_allfreq[freq_lpf_sele, :]
    # % Spectrum Normalization By Sum For Each Snapshot
    freq_time_prof = abs(freq_time_prof) / abs(freq_time_prof).sum(0)
    #
    # % Frequency Bin(Corresponding to FFT Results)
    freq_bin = np.r_[0 : freq_lpf_positive_max + 1, -1 * freq_lpf_negative_min : 0]
    #
    # % Store Doppler Velocity Spectrum
    if freq_time_prof.shape[1] >= doppler_spectrum.shape[1]:
        doppler_spectrum = freq_time_prof[:, : doppler_spectrum.shape[1]]
    else:
        doppler_spectrum = np.concatenate(
            [
                freq_time_prof,
                np.zeros(
                    (
                        doppler_spectrum.shape[0],
                        doppler_spectrum.shape[1] - freq_time_prof.shape[1],
                    )
                ),
            ]
        )
    doppler_spectrum[:] = np.fft.fftshift(doppler_spectrum, axes=0)
    return doppler_spectrum


def widar3_dfs(imax):
    csi_path = Path("/media/yk/linux_data/csi_dataset_survey/dataset/widar/mat_save_folder")
    ret = []
    for i, n0 in enumerate(sorted(csi_path.glob("*.mat"))):
        print(i)
        csi_data = scipy.io.loadmat(n0)["csi_data"]
        ret.append(dfs(csi_data.reshape(-1, 90)))
        if i > imax:
            break
    return ret


if __name__ == "__main__":

    d = widar3_dfs(9)

    # fig, ax = plt.subplots(3, 3)

    # for i, j in product(range(3), range(3)):
    #     start = time.time()
    #     dp, pos = jl_ex_plcr(dd)
    #     res = process_f1(dp, pos)
    #     print(time.time() - start)
    #
    # ax[i, j].contour(d[i * 3 + j], levels=30)

    #     plt.plot(res[:, 1], res[:, 0])
    # fig.tight_layout(pad=0)
    # plt.show()
    import plotly.graph_objects as go

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
