from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

from dataset_manager import CSI_ZARR_DATASET
from math_utils import complex_to_amp_pha
from widar_reinpl_series import gen_iir_filter, gen_fir_filter, conj_multi_csi
from widar_reinpl_series.re_inpl_filter import apply_csi_filter


def full_extent(ax, pad=0.0):
    from matplotlib.transforms import Bbox

    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)


def filter_fig_plot():
    csi_path = Path("dataset/c301_csi/csi")
    # csi_path = Path(
    #     "dataset/subset1_np_{'receiverid': (3,), 'face_orientation': (1, 2, 3), 'location': (5,)}/csi"
    # )
    d = CSI_ZARR_DATASET(csi_path)
    img_path = Path("dataset/images")
    samp_rate = 1000
    f_ = gen_fir_filter(samp_rate, 200)

    for i in range(22, 24):
        print(i)
        csi_data = d[i]["csi_data_raw"]

        csi_data_2 = f_(x=csi_data)
        csi_data_2 = conj_multi_csi(csi_data_2) * 60

        # csi_data_2 = conj_multi_csi(csi_data)
        # csi_data_2 = conj_multi_csi(csi_data_1) * 60

        amp1, pha1 = complex_to_amp_pha(csi_data)
        # amp2, pha2 = complex_to_amp_pha(csi_data_1)
        amp4, pha4 = complex_to_amp_pha(csi_data_2)

        amp1 = scipy.ndimage.uniform_filter(amp1, size=(55, 1, 1))
        # amp2 = scipy.ndimage.uniform_filter(amp2, size=(55, 1, 1))
        amp4 = scipy.ndimage.uniform_filter(amp4, size=(55, 1, 1))
        fig, ax = plt.subplots(2, 2)
        subcarrier_idx = 75

        ax[0, 0].set_xlabel("frame index")
        ax[0, 0].set_ylabel("CSI amplitude")
        line = ax[0, 0].plot(amp1[:, :, subcarrier_idx])
        ax[0, 0].legend(
            line, [f"subcarrier {subcarrier_idx} antenna {i}" for i in range(3)], loc=1,
        )
        ax[0, 1].set_xlabel("frame index")
        ax[0, 1].set_ylabel("CSI amplitude")
        line = ax[0, 1].plot(amp4[:, :, subcarrier_idx])
        ax[0, 1].legend(
            line, [f"subcarrier {subcarrier_idx} antenna {i}" for i in range(3)], loc=1,
        )
        ax[1, 0].set_xlabel("frame index")
        ax[1, 0].set_ylabel("Unwrapped CSI phase")
        line = ax[1, 0].plot(pha1[:, :, subcarrier_idx])
        ax[1, 0].legend(
            line, [f"subcarrier {subcarrier_idx} antenna {i}" for i in range(3)], loc=1,
        )
        ax[1, 1].set_xlabel("frame index")
        ax[1, 1].set_ylabel("Unwrapped CSI phase")
        line = ax[1, 1].plot(pha4[:, :, subcarrier_idx])
        ax[1, 1].legend(
            line, [f"subcarrier {subcarrier_idx} antenna {i}" for i in range(3)], loc=1,
        )

        fig.set_size_inches(18, 15)
        fig.set_dpi(300)
        # fig.savefig(img_path / f"fig{i}.svg", format="svg", bbox_inches="tight")

        extent = full_extent(ax[0, 0]).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(img_path / "0.png", bbox_inches=extent)
        extent = full_extent(ax[0, 1]).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(img_path / "1.png", bbox_inches=extent)
        extent = full_extent(ax[1, 0]).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(img_path / "2.png", bbox_inches=extent)
        extent = full_extent(ax[1, 1]).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(img_path / "3.png", bbox_inches=extent)

        plt.close(fig)

        fig, ax = plt.subplots(2, 2)
        packet_index = 1000

        ax[0, 0].set_xlabel("packet index")
        ax[0, 0].set_ylabel("CSI amplitude")
        line = ax[0, 0].plot(amp1[packet_index, :, :].T)
        ax[0, 0].legend(
            line, [f"subcarrier {subcarrier_idx} antenna {i}" for i in range(3)], loc=1,
        )
        ax[0, 1].set_xlabel("packet index")
        ax[0, 1].set_ylabel("CSI amplitude")
        line = ax[0, 1].plot(amp4[packet_index, :, :].T)
        ax[0, 1].legend(
            line, [f"subcarrier {subcarrier_idx} antenna {i}" for i in range(3)], loc=1,
        )
        ax[1, 0].set_xlabel("packet index")
        ax[1, 0].set_ylabel("Unwrapped CSI phase")
        line = ax[1, 0].plot(pha1[packet_index, :, :].T)
        ax[1, 0].legend(
            line, [f"subcarrier {subcarrier_idx} antenna {i}" for i in range(3)], loc=1,
        )
        ax[1, 1].set_xlabel("packet index")
        ax[1, 1].set_ylabel("Unwrapped CSI phase")
        line = ax[1, 1].plot(pha4[packet_index, :, :].T)
        ax[1, 1].legend(
            line, [f"subcarrier {subcarrier_idx} antenna {i}" for i in range(3)], loc=1,
        )

        fig.set_size_inches(3.5 * 5, 4 * 4)
        fig.set_dpi(200)
        # fig.savefig(img_path / f"fig{i}_.svg", format="svg", bbox_inches="tight")
        plt.close(fig)
        # plt.show()


def test():
    csi_path = Path("dataset/c301_csi/csi")
    csi_path = Path(
        "dataset/subset1_np_{'receiverid': (3,), 'face_orientation': (1, 2, 3), 'location': (5,)}/csi"
    )
    d = CSI_ZARR_DATASET(csi_path)

    samp_rate = 1000
    f_fir = gen_fir_filter(samp_rate)
    f_fir2 = gen_fir_filter(samp_rate, irlen=150)
    f_iir = gen_iir_filter(samp_rate)
    for i in range(len(d)):
        print(i)
        csi_data = d[i]["csi_data_raw"]
        csi_data = apply_csi_filter(x=csi_data, filter=f_fir2)
        csi_data_1, idx1 = conj_multi_csi(csi_data, 15)
        csi_data_2, idx2 = conj_multi_csi(csi_data, 50)
        csi_data_3, idx3 = conj_multi_csi(csi_data, 100)

        # csi_data = conj_multi_csi(csi_data)
        # csi_data_1 = conj_multi_csi(apply_csi_filter(x=csi_data, filter=f_fir))
        # csi_data_2 = conj_multi_csi(apply_csi_filter(x=csi_data, filter=f_fir2))
        # csi_data_3 = conj_multi_csi(apply_csi_filter(x=csi_data, filter=f_iir))

        # print(np.abs(csi_data_1 - csi_data_2).sum())

        amp1, pha1 = complex_to_amp_pha(csi_data)
        amp2, pha2 = complex_to_amp_pha(csi_data_1)
        amp3, pha3 = complex_to_amp_pha(csi_data_2)
        amp4, pha4 = complex_to_amp_pha(csi_data_3)

        # amp1, pha1 = np.abs(csi_data), np.angle(csi_data)
        # amp2, pha2 = np.abs(csi_data_1), np.angle(csi_data_1)
        # amp3, pha3 = np.abs(csi_data_2), np.angle(csi_data_2)
        # amp4, pha4 = np.abs(csi_data_3), np.angle(csi_data_3)

        for a, p in [(amp1, pha1), (amp2, pha2), (amp3, pha3), (amp4, pha4)]:
            print(a.max(), a.min(), p.max(), p.min())

        subcarrier_idx = 25
        set_d = {"s": 2, "alpha": 0.005}
        set_line = {"linewidth": 0.7, "alpha": 1}

        plt.subplot(4, 4, 1)
        line = plt.plot(amp1[:, :, subcarrier_idx], **set_line)
        plt.legend(
            line, [f"subcarrier {subcarrier_idx} antenna {i}" for i in range(3)], loc=1,
        )
        plt.subplot(4, 4, 2)
        plt.plot(amp2[:, :, subcarrier_idx], **set_line)
        plt.subplot(4, 4, 3)
        plt.plot(amp3[:, :, subcarrier_idx], **set_line)
        plt.subplot(4, 4, 4)
        plt.plot(amp4[:, :, subcarrier_idx], **set_line)
        plt.subplot(4, 4, 5)
        plt.plot(pha1[:, :, subcarrier_idx], **set_line)
        plt.subplot(4, 4, 6)
        plt.plot(pha2[:, :, subcarrier_idx], **set_line)
        plt.subplot(4, 4, 7)
        plt.plot(pha3[:, :, subcarrier_idx], **set_line)
        plt.subplot(4, 4, 8)
        plt.plot(pha4[:, :, subcarrier_idx], **set_line)

        plt.subplot(4, 4, 9)
        plt.scatter(csi_data.ravel().real, csi_data.ravel().imag, **set_d)
        plt.subplot(4, 4, 10)
        plt.scatter(csi_data_1.ravel().real, csi_data_1.ravel().imag, **set_d)
        plt.subplot(4, 4, 11)
        plt.scatter(
            csi_data_2.ravel().real, csi_data_2.ravel().imag, **set_d,
        )
        plt.subplot(4, 4, 12)
        plt.scatter(
            csi_data_3.ravel().real, csi_data_3.ravel().imag, **set_d,
        )
        plt.subplot(4, 4, 14)
        plt.semilogy(idx1, **set_line)
        plt.subplot(4, 4, 15)
        plt.semilogy(idx2, **set_line)
        plt.subplot(4, 4, 16)
        plt.semilogy(idx3, **set_line)

        plt.show()


def final():
    csi_path = Path("dataset/c301_csi/csi")
    # csi_path = Path(
    #     "dataset/subset1_np_{'receiverid': (3,), 'face_orientation': (1, 2, 3), 'location': (5,)}/csi"
    # )
    d = CSI_ZARR_DATASET(csi_path)
    img_path = Path("dataset/images")
    samp_rate = 1000
    f_ = gen_fir_filter(samp_rate, 200)
    i = 23

    csi_data = d[i]["csi_data_raw"]

    csi_data_2 = f_(x=csi_data)
    csi_data_2 = conj_multi_csi(csi_data_2) * 60

    amp1, pha1 = complex_to_amp_pha(csi_data)
    amp4, pha4 = complex_to_amp_pha(csi_data_2)

    amp1 = scipy.ndimage.uniform_filter(amp1, size=(55, 1, 1))
    amp4 = scipy.ndimage.uniform_filter(amp4, size=(55, 1, 1))

    subcarrier_idx = 75
    fig, ax = plt.subplots()
    ax.set_xlabel("frame index")
    ax.set_ylabel("CSI amplitude")
    line = ax.plot(amp1[:, :, subcarrier_idx])
    ax.legend(
        line, [f"subcarrier {subcarrier_idx} antenna {i}" for i in range(3)], loc=1,
    )
    fig.set_size_inches(6.5, 5)
    fig.set_dpi(200)
    fig.savefig(img_path / f"fig{i}_0.svg", format="svg", bbox_inches="tight")

    fig, ax = plt.subplots()
    ax.set_xlabel("frame index")
    ax.set_ylabel("CSI amplitude")
    line = ax.plot(amp4[:, :, subcarrier_idx])
    ax.legend(
        line, [f"subcarrier {subcarrier_idx} antenna {i}" for i in range(3)], loc=1,
    )
    fig.set_size_inches(6.5, 5)
    fig.set_dpi(200)
    fig.savefig(img_path / f"fig{i}_1.svg", format="svg", bbox_inches="tight")

    fig, ax = plt.subplots()
    ax.set_xlabel("frame index")
    ax.set_ylabel("Unwrapped CSI phase")
    line = ax.plot(pha1[:, :, subcarrier_idx])
    ax.legend(
        line, [f"subcarrier {subcarrier_idx} antenna {i}" for i in range(3)], loc=1,
    )
    fig.set_size_inches(6.5, 5)
    fig.set_dpi(200)
    fig.savefig(img_path / f"fig{i}_2.svg", format="svg", bbox_inches="tight")

    fig, ax = plt.subplots()
    ax.set_xlabel("frame index")
    ax.set_ylabel("Unwrapped CSI phase")
    line = ax.plot(pha4[:, :, subcarrier_idx])
    ax.legend(
        line, [f"subcarrier {subcarrier_idx} antenna {i}" for i in range(3)], loc=1,
    )
    fig.set_size_inches(6.5, 5)
    fig.set_dpi(200)
    fig.savefig(img_path / f"fig{i}_3.svg", format="svg", bbox_inches="tight")


def plot2svg(data, name, size, xlabel, ylabel, legend):
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    line = ax.plot(data)
    if legend is not None:
        ax.legend(
            line, legend, loc=1,
        )
    fig.set_size_inches(*size)
    fig.set_dpi(200)
    fig.savefig(name, format="svg", bbox_inches="tight")
    plt.close(fig)


import tqdm


def zxf2():
    csi_path = Path("dataset/c301_csi/csi")

    d = CSI_ZARR_DATASET(csi_path)
    l_d = d.get_label_dict()
    img_path = Path("dataset/images")

    id = reduce(
        np.logical_and,
        [
            # l_d["csi_label_act"] == 2,
            # l_d["csi_label_loc"] == 1,
            l_d["csi_label_user"] == 1,
            l_d["csi_label_env"] == 1,
        ],
    )
    id = np.where(id)[0]

    subcarrier_idx = 66
    samp_rate = 1000
    size = (6.5, 6)
    timesize = slice(200, 400)
    f_fir = gen_fir_filter(samp_rate)
    f_fir2 = gen_fir_filter(samp_rate, irlen=150)
    f_iir = gen_iir_filter(samp_rate)
    for i in tqdm.tqdm(id):
        data = d[i]["csi_data_raw"]
        dic = {}
        amp1, pha1 = complex_to_amp_pha(data)
        pha0 = np.angle(data)
        data2 = apply_csi_filter(data, f_fir2)
        amp2, pha2 = complex_to_amp_pha(data2)
        dic["pha0"] = pha0
        dic["pha1"] = pha1
        dic["pha2"] = pha2
        dic["amp1"] = amp1
        dic["amp2"] = amp2
        np.save(img_path / f"{i}.npy", dic)


def zxf():
    csi_path = Path("dataset/c301_csi/csi")

    d = CSI_ZARR_DATASET(csi_path)
    l_d = d.get_label_dict()
    img_path = Path("dataset/images")

    id = reduce(
        np.logical_and,
        [
            # l_d["csi_label_act"] == 2,
            # l_d["csi_label_loc"] == 1,
            l_d["csi_label_user"] == 1,
            l_d["csi_label_env"] == 1,
        ],
    )
    id = np.where(id)[0]

    subcarrier_idx = 66
    samp_rate = 1000
    size = (6.5, 6)
    timesize = slice(200, 400)
    f_fir = gen_fir_filter(samp_rate)
    f_fir2 = gen_fir_filter(samp_rate, irlen=150)
    f_iir = gen_iir_filter(samp_rate)
    for i in tqdm.tqdm(id):
        data = d[i]["csi_data_raw"]
        amp1, pha1 = complex_to_amp_pha(data)
        pha0 = np.angle(data)
        data2 = apply_csi_filter(data, f_fir2)
        amp2, pha2 = complex_to_amp_pha(data2)

        plot2svg(
            amp1[timesize, 0, subcarrier_idx],
            img_path / f"id_{i}_amp1.svg",
            size,
            "package index",
            "CSI amplitude",
            None,
        )
        plot2svg(
            pha0[timesize, 0, subcarrier_idx],
            img_path / f"id_{i}_pha1.svg",
            size,
            "package index",
            "CSI phase",
            None,
        )
        plot2svg(
            amp2[timesize, 0, subcarrier_idx],
            img_path / f"id_{i}_amp2.svg",
            size,
            "package index",
            "CSI amplitude",
            None,
        )

        plot2svg(
            pha1[timesize, 0, subcarrier_idx],
            img_path / f"id_{i}_pha2.svg",
            size,
            "package index",
            "CSI phase",
            None,
        )

        plot2svg(
            pha2[timesize, 0, subcarrier_idx],
            img_path / f"id_{i}_pha3.svg",
            size,
            "package index",
            "CSI phase",
            None,
        )

    pass


# test()
# filter_fig_plot()
# final()
# zxf()
zxf2()
