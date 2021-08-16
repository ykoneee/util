import numba
import numpy as np

from math_utils import split_array_bychunk


@numba.njit(cache=True)
def conj_multi_antenna_pair(x1, x2):
    amp_x1 = np.abs(x1)
    amp_x2 = np.abs(x2)

    tmp = amp_x1.ravel()
    alpha = np.min(tmp[tmp != 0]) - 1e-1
    beta = 1000 * alpha

    x1_adj = (amp_x1 - alpha) * np.exp(1j * np.angle(x1))

    x2_adj = (amp_x2 + beta) * np.exp(1j * np.angle(x2))

    x_conj = x1_adj * x2_adj.conj()

    x_conj = x_conj - x_conj.mean()

    x_conj = x_conj / 1000
    return x_conj


def antenna_choose_widar(x_in):
    csi_mean = np.abs(x_in).mean(0)
    csi_var = np.sqrt(np.abs(x_in).var(0, ddof=1))
    csi_mean_var_ratio = csi_mean / csi_var
    idx = np.argmax(csi_mean_var_ratio.mean(1))
    return idx


def antenna_choose_widar_modified(x_in):
    csi_mean = np.abs(x_in).mean(0)
    csi_var = np.sqrt(np.abs(x_in).var(0, ddof=1))
    csi_mean_var_ratio = csi_mean / csi_var

    csi_mean_var_ratio = np.tile(csi_mean_var_ratio.mean(1), (3, 1))
    csi_mean_var_ratio[0, 0] = -1
    csi_mean_var_ratio[1, 1] = -1
    csi_mean_var_ratio[2, 2] = -1
    idx = np.argmax(csi_mean_var_ratio, axis=1)
    return idx


@numba.njit(cache=True)
def antenna_choose_base_jit(x_l_in):
    window_n = len(x_l_in)
    antenna_n = x_l_in[0].shape[1]
    subc_n = x_l_in[0].shape[2]
    ret = np.zeros((window_n, antenna_n, subc_n))
    for w_idx in range(window_n):
        for antenna_idx in range(antenna_n):
            for subc_idx in range(subc_n):
                csi_mean = np.abs(x_l_in[w_idx][:, antenna_idx, subc_idx]).mean()
                csi_var = np.var(np.abs(x_l_in[w_idx][:, antenna_idx, subc_idx]))
                csi_mean_var_ratio = csi_mean / (csi_var + 1e-6)
                ret[w_idx, antenna_idx, subc_idx] += csi_mean_var_ratio
    return ret


def antenna_choose_base(x_l_in):
    window_n = len(x_l_in)
    antenna_n = x_l_in[0].shape[1]
    subc_n = x_l_in[0].shape[2]
    ret = np.zeros((window_n, antenna_n, subc_n))
    for w_idx in range(window_n):
        for antenna_idx in range(antenna_n):
            for subc_idx in range(subc_n):
                csi_mean = np.abs(x_l_in[w_idx][:, antenna_idx, subc_idx]).mean()
                csi_var = np.abs(x_l_in[w_idx][:, antenna_idx, subc_idx]).var()

                csi_mean_var_ratio = csi_mean / (csi_var + 1e-6)
                ret[w_idx, antenna_idx, subc_idx] += csi_mean_var_ratio
    return ret


def antenna_choose_mutrack(x_l_in):
    return np.argmax(antenna_choose_base(x_l_in).sum(axis=1))


def conj_multi_csi(x_in, chunk_size=50):
    idx = antenna_choose_widar_modified(x_in)

    # print(idx)
    idx = np.array([1, 2, 0])
    # idx = np.array([2, 0, 1])

    xl, xl_res = split_array_bychunk(x_in, chunk_size, cat_residual=False)
    xl = np.stack(xl)

    mean_var_ratio = antenna_choose_base_jit(xl).mean(axis=2)

    if xl_res is not None and len(xl_res) > 2:
        mean_var_ratio_res = antenna_choose_base_jit(xl_res[None, ...]).mean(axis=2)
        mean_var_ratio = np.concatenate([mean_var_ratio, mean_var_ratio_res])

    antenna_idx = np.tile(mean_var_ratio.copy(), (3, 1, 1))

    antenna_idx[0, :, 0] = -1
    antenna_idx[1, :, 1] = -1
    antenna_idx[2, :, 2] = -1
    antenna_idx = np.argmax(antenna_idx, axis=2)

    # x_res = []
    # for i in range(len(xl)):
    #     x1 = conj_multi_antenna_pair(xl[i, :, 0, :], xl[i, :, antenna_idx[0, i], :])
    #     x2 = conj_multi_antenna_pair(xl[i, :, 1, :], xl[i, :, antenna_idx[1, i], :])
    #     x3 = conj_multi_antenna_pair(xl[i, :, 2, :], xl[i, :, antenna_idx[2, i], :])
    #     x_res.append(np.stack([x1, x2, x3], axis=1))
    #
    # if xl_res is not None and len(xl_res) > 2:
    #     x1 = conj_multi_antenna_pair(xl_res[:, 0, :], xl_res[:, antenna_idx[0, -1], :])
    #     x2 = conj_multi_antenna_pair(xl_res[:, 1, :], xl_res[:, antenna_idx[1, -1], :])
    #     x3 = conj_multi_antenna_pair(xl_res[:, 2, :], xl_res[:, antenna_idx[2, -1], :])
    #     x_res.append(np.stack([x1, x2, x3], axis=1))

    x_res = []
    for i in range(len(xl)):
        x1 = conj_multi_antenna_pair(xl[i, :, 0, :], xl[i, :, idx[0], :])
        x2 = conj_multi_antenna_pair(xl[i, :, 1, :], xl[i, :, idx[1], :])
        x3 = conj_multi_antenna_pair(xl[i, :, 2, :], xl[i, :, idx[2], :])
        x_res.append(np.stack([x1, x2, x3], axis=1))

    if xl_res is not None and len(xl_res) > 2:
        x1 = conj_multi_antenna_pair(xl_res[:, 0, :], xl_res[:, idx[0], :])
        x2 = conj_multi_antenna_pair(xl_res[:, 1, :], xl_res[:, idx[1], :])
        x3 = conj_multi_antenna_pair(xl_res[:, 2, :], xl_res[:, idx[2], :])
        x_res.append(np.stack([x1, x2, x3], axis=1))

    x_res = np.concatenate(x_res)

    return x_res, mean_var_ratio


def speed_test():
    import time

    n = 100
    xl = [
        np.random.rand(1800, 3, 114) + 1j * np.random.rand(1800, 3, 114)
        for i in range(n)
    ]
    t1 = time.time()
    for i, x in enumerate(xl):
        print(i)
        r = conj_multi_csi(x)
    print(time.time() - t1)


if __name__ == "__main__":
    speed_test()
