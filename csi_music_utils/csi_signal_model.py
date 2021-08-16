# print("in csi_signal_model.py")
import numpy as np
import torch

from . import (
    channel_center_freq_5ghz,
    subcarrier_spacing,
    subcarrier_index_dict,
    optimal_antenna_d,
    PI,
    C,
)
from math_utils import randcn


def Freq_subcarrier(wifi_channel, subcarrier_num):
    # return: list of sub_carrier freq
    if subcarrier_num == 30:
        s_index = subcarrier_index_dict["20MHZ"]["Ng2-30"]
        return channel_center_freq_5ghz(wifi_channel) + subcarrier_spacing * s_index
    elif subcarrier_num == 114:
        s_index = subcarrier_index_dict["40MHZ"]["Ng1-114"]
        return channel_center_freq_5ghz(wifi_channel) + subcarrier_spacing * s_index
    else:
        assert False  # Subcarrier num don't support!!


def Delta_t_sample_packet(samples_len, sample_freq):
    return np.arange(samples_len) * (1 / sample_freq)


def Antenna_d(antenna_num, d):
    return np.arange(antenna_num) * d


class CSIBaseModel:
    def __init__(self, sample_len, channel, sample_freq, subcarrier_num):
        self.antenna_num = 3
        self.sample_len = sample_len
        self.channel = channel
        self.sample_freq = sample_freq
        self.subcarrier_num = subcarrier_num
        self.subcarrier_spacing = subcarrier_spacing
        self.freq = channel_center_freq_5ghz(self.channel)
        self.d = optimal_antenna_d(self.channel)

        self.delta_t_array = Delta_t_sample_packet(self.sample_len, self.sample_freq)
        self.d_array = Antenna_d(self.antenna_num, self.d)
        self.f_array = Freq_subcarrier(self.channel, self.subcarrier_num)


class CSIGeneraterModel(CSIBaseModel):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.multi_path_num_true = 6
        self.multi_param = (
            np.array(
                [
                    [1, 0.55, 0.6, 0.7, 0.8, 0.9],
                    [123, 234, 345, 456, 567, 678],
                    [3, -6, -2, 2, 6, 10],
                    [66, 130, 25, 50, 85, 110],
                ],
            )
            .T.ravel()
            .view(
                dtype=[
                    ("attenuation", "float"),
                    ("tof", "float"),
                    ("dfs", "float"),
                    ("aoa", "float"),
                ],
            )
        )  # attenuation, tof, dop_v, aoa
        self.multi_param["tof"] *= 1e-9
        self.multi_param["aoa"] = self.multi_param["aoa"] * PI / 180
        self.coherence = False

    def a_(self, tof, dfs, aoa):
        return np.exp(
            -1j
            * 2
            * PI
            * (
                self.f_array[None, None, :]
                * (tof + SteeringVector_aoa_(self.d_array, aoa)[None, :, None])
                + SteeringVector_dfs_(
                    self.delta_t_array,
                    dfs,
                )[:, None, None]
            )
        )

    def gen_signal(self):

        if self.coherence:
            cov = np.random.normal(
                size=(self.multi_path_num_true, self.multi_path_num_true)
            )
            cov = cov.T @ cov
        else:
            cov = np.diag(np.ones(self.multi_path_num_true))

        s = cov @ randcn((self.multi_path_num_true, self.sample_len))

        As = sum(
            [
                s[i][:, None, None]
                * self.a_(
                    self.multi_param["tof"][i],
                    self.multi_param["dfs"][i],
                    self.multi_param["aoa"][i],
                )
                for i in range(self.multi_path_num_true)
            ]
        )

        noise = randcn(As.shape) * 0.05
        X = As + noise

        X = X.astype(np.complex64)

        return X


# def SteeringVector_aoa(f, dist, angle_of_arrival):
#     return np.exp(-1j * 2 * PI * f * dist * np.cos(angle_of_arrival) / C)
#
#
# def SteeringVector_dfs(f, delta_t, doppler_v):
#     return np.exp(1j * 2 * PI * f * doppler_v * delta_t / C)


# def SteeringVector_tof(f, tof):
#     return np.exp(-1j * 2 * PI * f * tof)


def SteeringVector_aoa_(dist, angle_of_arrival):
    return dist * np.cos(angle_of_arrival) / C


def SteeringVector_aoa_torch(dist, angle_of_arrival):
    return dist * torch.cos(angle_of_arrival) / C


def SteeringVector_dfs_(delta_t, dfs):
    return dfs * delta_t


if __name__ == "__main__":
    pass
