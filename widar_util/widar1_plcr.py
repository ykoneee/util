import numpy as np
import numba


@numba.njit(cache=True)
def extract_plcr(spectrum, step_size=3):
    max_freq = spectrum.shape[0]
    max_time = spectrum.shape[1]
    __pad = np.ones((step_size, max_time)) * -1
    spectrum = np.concatenate((__pad, spectrum, __pad), axis=0)
    # spectrum = np.pad(spectrum, ((step_size, step_size), (0, 0)), mode="constant", constant_values=-1)
    dp_c = np.ones_like(spectrum) * -1
    dp_c[:, -1] = spectrum[:, -1]
    pos_c = np.ones_like(spectrum, dtype=np.int16)
    pos_c[:, -1] = np.arange(max_freq + 2 * step_size)

    # __t_range = range(max_time - 1)[::-1]
    for cur_t in range(max_time - 1 - 1, -1, -1):
        for cur_freq in range(step_size, max_freq + step_size):
            pos = np.arange(step_size * 2 + 1) - step_size + cur_freq
            value = dp_c[pos, cur_t + 1]
            max_idx = np.argmax(value)
            dp_c[cur_freq, cur_t] = spectrum[cur_freq, cur_t] + value[max_idx]
            pos_c[cur_freq, cur_t] = pos[max_idx]
    dp_c = dp_c[step_size:-step_size, :]
    pos_c = pos_c[step_size:-step_size, :] - step_size  # cause we padding pos_c
    return process_(dp_c, pos_c)


@numba.njit(cache=True)
def process_(dp, pos):
    sorted_index = np.argsort(-dp[:, 0])
    res = np.ones_like(dp, dtype=np.int16)
    t_max = dp.shape[1]
    for i in range(len(sorted_index)):
        last_pos = sorted_index[i]
        res[i, 0] = last_pos
        for idx_t in range(1, t_max):
            last_pos = pos[last_pos, idx_t]
            res[i, idx_t] = last_pos
    return res
