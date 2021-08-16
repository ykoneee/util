# print("in csi_music.py")
import time

import cupy
import numpy as np
import torch

from . import PI
from . import (
    SteeringVector_dfs_,
    SteeringVector_aoa_torch,
    CSIGeneraterModel,
)
from . import rolling_window
from math_utils import split_array_bychunk


def convert2tensor(x):
    if x.dtype == np.float64:
        x = x.astype(np.float32)
    if x.dtype == np.complex128:
        x = x.astype(np.complex64)
    return torch.as_tensor(x, device="cuda")


class NDMUSIC:
    def __init__(
        self,
        signal_model,
        window,
        sspace_grid,
        tof_sspace_max,
        dfs_sspace_max,
        aoa_sspace_max,
        gpu_data_slice_window=(10, 10, 10),
    ):
        self.signal_model = signal_model
        self.multi_path_num_solve = 8
        self.sspace_grid = sspace_grid
        self.tof_sspace_max = tof_sspace_max
        self.dfs_sspace_max = dfs_sspace_max
        self.aoa_sspace_max = aoa_sspace_max
        self.smooth = True
        self.gpu_data_slice_window = gpu_data_slice_window
        subcarrier_num = self.signal_model.subcarrier_num
        assert (
            (self.signal_model.sample_len - window[0]) > 0
            and (self.signal_model.antenna_num - window[1]) > 0
            and (subcarrier_num - window[2]) > 0
        )
        self.smooth_window = (
            (
                window[0],
                window[1],
                window[2],
            )
            if self.smooth
            else (
                self.signal_model.sample_len,
                self.signal_model.antenna_num,
                subcarrier_num,
            )
        )

        self.delta_t_array = convert2tensor(
            self.signal_model.delta_t_array[
                None,
                None,
                None,
                : self.smooth_window[0],
                None,
                None,
            ]
        )
        self.dist_array = convert2tensor(
            self.signal_model.d_array[
                None,
                None,
                None,
                None,
                : self.smooth_window[1],
                None,
            ]
        )
        self.f_array = convert2tensor(
            self.signal_model.f_array[
                None,
                None,
                None,
                None,
                None,
                : self.smooth_window[2],
            ]
        )

    def p_batch_torch(self, UnUnH, tof_sspace, dfs_sspace, aoa_sspace):
        UnUnH = convert2tensor(UnUnH)
        tof_sspace_len = tof_sspace.shape[0]
        dfs_sspace_len = dfs_sspace.shape[0]
        aoa_sspace_len = aoa_sspace.shape[0]
        ret = convert2tensor(
            np.zeros(
                (tof_sspace_len, dfs_sspace_len, aoa_sspace_len), dtype=np.complex64
            )
        )

        tof_range_list = split_array_bychunk(
            np.arange(tof_sspace_len), self.gpu_data_slice_window[0]
        )
        dfs_range_list = split_array_bychunk(
            np.arange(dfs_sspace_len), self.gpu_data_slice_window[1]
        )
        aoa_range_list = split_array_bychunk(
            np.arange(aoa_sspace_len), self.gpu_data_slice_window[2]
        )

        dfs_sspace = convert2tensor(
            dfs_sspace[
                None,
                :,
                None,
                None,
                None,
                None,
            ]
        )
        tof_sspace = convert2tensor(
            tof_sspace[
                :,
                None,
                None,
                None,
                None,
                None,
            ]
        )
        aoa_sspace = convert2tensor(
            aoa_sspace[
                None,
                None,
                :,
                None,
                None,
                None,
            ]
        )

        def a_batch_(tof_r, dfs_r, aoa_r):
            return torch.exp(
                -1j
                * 2
                * PI
                * (
                    self.f_array
                    * (
                        tof_sspace[
                            tof_r,
                            :,
                            :,
                            :,
                            :,
                            :,
                        ]
                        + SteeringVector_aoa_torch(
                            self.dist_array,
                            aoa_sspace[
                                :,
                                :,
                                aoa_r,
                                :,
                                :,
                                :,
                            ],
                        )
                    )
                    + SteeringVector_dfs_(
                        self.delta_t_array,
                        dfs_sspace[
                            :,
                            dfs_r,
                            :,
                            :,
                            :,
                            :,
                        ],
                    )
                )
            )

        for tof_range in tof_range_list:
            for dfs_range in dfs_range_list:
                for aoa_range in aoa_range_list:
                    a_ = a_batch_(tof_range, dfs_range, aoa_range)
                    a_ = a_.reshape(
                        *a_.shape[:3],
                        -1,
                        1,
                    )
                    a_H = torch.transpose(a_.conj(), 3, 4)
                    P_ = torch.squeeze(torch_jit_f(a_H, UnUnH, a_))

                    ret[
                        tof_range[:, None, None],
                        dfs_range[None, :, None],
                        aoa_range[None, None, :],
                    ] = P_
        return ret.real

    def P_3D_divided(self, Un):

        tof_n = self.sspace_grid[0]
        dfs_n = self.sspace_grid[1]
        aoa_n = self.sspace_grid[2]
        self.tof_searchspace = np.linspace(
            0, self.tof_sspace_max / self.signal_model.subcarrier_spacing, tof_n
        )
        self.dfs_searchspace = np.linspace(
            -self.dfs_sspace_max, self.dfs_sspace_max, dfs_n
        )
        self.aoa_searchspace = np.linspace(0, self.aoa_sspace_max, aoa_n) * PI / 180
        # print("search space:")
        # print(
        #     f"    tof:{self.tof_searchspace.min()*1e9,self.tof_searchspace.max()*1e9}"
        # )
        # print(f"    dfs:{self.dfs_searchspace.min(),self.dfs_searchspace.max()}")
        # print(
        #     f"    aoa:{self.aoa_searchspace.min()/ PI * 180,self.aoa_searchspace.max()/ PI * 180}"
        # )

        UnUnH = Un @ Un.conj().T
        with torch.no_grad():
            P = self.p_batch_torch(
                UnUnH,
                self.tof_searchspace,
                self.dfs_searchspace,
                self.aoa_searchspace,
            )

        return 1.0 / P.cpu().numpy()

    def P_3D_root(self, Un):
        UnH = Un.conj().T
        UnUnH = np.asarray(np.matmul(Un, UnH))
        m = UnUnH.shape[0]
        coeff = np.zeros((m - 1,), dtype=np.complex_)
        for i in range(1, m):
            coeff[i - 1] += np.sum(np.diag(UnUnH, i))
        coeff = np.hstack((coeff[::-1], np.sum(np.diag(UnUnH)), coeff.conj()))
        # Find the roots of the polynomial.
        z = np.roots(coeff)
        print(len(z))
        nz = len(z)
        mask = np.ones((nz,), dtype=np.bool_)
        for i in range(nz):
            absz = abs(z[i])
            if absz > 1.0:
                # Outside the unit circle.
                mask[i] = False
            elif absz == 1.0:
                # On the unit circle. Need to find the closest point and remove
                # it.
                idx = -1
                dist = np.inf
                for j in range(nz):
                    if j != i and mask[j]:
                        cur_dist = abs(z[i] - z[j])
                        if cur_dist < dist:
                            dist = cur_dist
                            idx = j
                if idx < 0:
                    raise RuntimeError(
                        "Unpaired point found on the unit circle, which is impossible."
                    )
                mask[idx] = False

        z = z[mask]
        sorted_indices = np.argsort(1.0 - np.abs(z))

        print(len(z))
        z = z[sorted_indices[: self.multi_path_num_solve]]

        Un1, Un2 = (
            Un[: self.multi_path_num_solve, :],
            Un[self.multi_path_num_solve :, :],
        )
        c = Un1 @ np.linalg.inv(Un2)[:, 0]

        root = -c  # FIX

    def signal_3D_smooth(self, X):

        X = rolling_window(X, window=self.smooth_window)
        # X = X[::2, :, ::2, ...]
        # X = X.reshape(-1, *self.smooth_window)[::2, :, ::2]
        X = X.reshape(-1, *self.smooth_window)
        X = X.transpose(1, 2, 3, 0)
        X = np.ascontiguousarray(X)

        X = X.reshape(-1, X.shape[3])
        return X

    def check_smooth_shape(self):
        print(
            f"ND MUSIC Calcution shape: X:{self.signal_model.sample_len},3,{self.signal_model.subcarrier_num} "
            f"X_windowed:{np.prod(self.smooth_window)},{np.prod([(self.signal_model.sample_len - self.smooth_window[0] + 1), (3 - self.smooth_window[1] + 1), (self.signal_model.subcarrier_num - self.smooth_window[2] + 1)])}"
        )

    def estimate(self, X, verbose=False):
        if isinstance(X, np.ndarray):
            X = X.astype(np.complex64)
        if verbose:
            print(f"X shape:{X.shape} X type:{X.dtype}")
        X = cupy.asarray(X)
        R = X @ X.conj().T / self.signal_model.sample_len
        if verbose:
            print(f"X*XH:R shape:{R.shape}")
            # print(f"R rank:{cupy.linalg.matrix_rank(R)}")

        eigvals, eigvecs = cupy.linalg.eigh(R)

        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        Us, Un = (
            eigvecs[:, : self.multi_path_num_solve],
            eigvecs[:, self.multi_path_num_solve :],
        )
        if verbose:
            print(f"Un size:{Un.shape}")

        P = self.P_3D_divided(Un)
        if verbose:
            print("estimate done")
        return P


# @jax.jit
def torch_jit_f(a, b, c):
    return torch.matmul(torch.matmul(a, b), c)


def estimate_idx_convert(estimater, idx):
    return np.array(
        [
            estimater.tof_searchspace[idx[0]] * 1e9,
            estimater.dfs_searchspace[idx[1]],
            estimater.aoa_searchspace[idx[2]] * 180 / PI,
        ]
    )


def speed_test():
    w = (75, 2, 40)
    sspace_grid = (30, 30, 30)
    tof_sspace_max = 0.5
    dfs_sspace_max = 25
    aoa_sspace_max = 180
    model = CSIGeneraterModel(
        sample_len=100, channel=157, sample_freq=1000, subcarrier_num=114
    )

    estimater = NDMUSIC(
        model,
        window=w,
        sspace_grid=sspace_grid,
        tof_sspace_max=tof_sspace_max,
        dfs_sspace_max=dfs_sspace_max,
        aoa_sspace_max=aoa_sspace_max,
    )
    estimater.check_smooth_shape()
    t0 = time.time()
    for i in range(15):
        print(i)
        X = model.gen_signal()
        print(X.shape)
        X = estimater.signal_3D_smooth(X)
        print(X.shape)
        P = estimater.estimate(X)
    t = (time.time() - t0) / 15 * 10000 / 3600
    print(f"10000 sample / {t:.3f} hours")


if __name__ == "__main__":
    speed_test()
    pass
