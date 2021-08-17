import numpy as np
from scipy import constants

C = constants.speed_of_light
PI = constants.pi
subcarrier_spacing = 312.5e3
subcarrier_spacing_ng2 = 125e3


def channel_center_freq_2p4ghz(channel):
    return (2407 + 5 * channel) * 1e6


def channel_center_freq_5ghz(channel):
    return (5000 + 5 * channel) * 1e6


def f2c_5ghz(f):
    return (f * 1e-6 - 5000) / 5


def optimal_antenna_d(channel):
    return C / channel_center_freq_5ghz(channel) / 2


subcarrier_index_dict = {  # https://ieeexplore.ieee.org/abstract/document/9051665
    "20MHZ": {
        "Ng1-56": np.concatenate(
            [np.arange(-28, -1, 1), [-1, 1], np.arange(2, 28, 1), [28]]
        ),
        "Ng2-30": np.concatenate(
            [np.arange(-28, -1, 2), [-1, 1], np.arange(3, 28, 2), [28]]
        ),
        "Ng4-16": np.concatenate(
            [np.arange(-28, -1, 4), [-1, 1], np.arange(5, 28, 4), [28]]
        ),
    },
    "40MHZ": {
        "Ng1-114": np.concatenate(
            [np.arange(-58, -2, 1), [-2, 2], np.arange(3, 59, 1)]
        ),
        "Ng2-58": np.concatenate([np.arange(-58, -2, 2), [-2, 2], np.arange(4, 59, 2)]),
        "Ng4-30": np.concatenate([np.arange(-58, -2, 4), [-2, 2], np.arange(6, 59, 4)]),
    },
}

resample_subcarrier_idx_114_30 = [
    np.flatnonzero(x == subcarrier_index_dict["40MHZ"]["Ng1-114"])
    for x in subcarrier_index_dict["20MHZ"]["Ng2-30"]
]
resample_subcarrier_idx_114_30[14] = [56]
resample_subcarrier_idx_114_30[15] = [57]
resample_subcarrier_idx_114_30 = np.array(resample_subcarrier_idx_114_30).flatten()
