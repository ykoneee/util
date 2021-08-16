def extract_plcr(sp):
    max_freq = sp.shape[0] - 1
    max_time = sp.shape[1] - 1

    dp_c = np.ones_like(sp) * -1
    dp_c[:, max_time] = sp[:, max_time]
    pos_c = np.ones((*sp.shape, 2), dtype=int) * -1

    def argmax(x):
        _idx = range(len(x))
        _x = max(zip(x, _idx))
        return _x

    for cur_t in range(max_time)[::-1]:
        for cur_freq in range(max_freq):
            if cur_freq == 0:
                pos = [(cur_freq, cur_t + 1), (cur_freq + 1, cur_t + 1)]
            elif cur_freq == max_freq:
                pos = [(cur_freq, cur_t + 1), (cur_freq - 1, cur_t + 1)]
            else:
                pos = [
                    (cur_freq - 1, cur_t + 1),
                    (cur_freq, cur_t + 1),
                    (cur_freq + 1, cur_t + 1),
                ]
            value = [dp_c[p[0], p[1]] for p in pos]
            max_v, max_idx = argmax(value)
            dp_c[cur_freq, cur_t] = sp[cur_freq, cur_t] + value[max_idx]
            pos_c[cur_freq, cur_t] = pos[max_idx]

    return dp_c, pos_c


def process_f1(dp, pos):
    last_pos = np.array([np.argmax(dp[:, 0]), 0])
    pos -= 1
    res = [last_pos]
    # print(dp.shape)
    while last_pos[1] != -2:
        last_pos = pos[last_pos[0], last_pos[1]]
        res.append(last_pos)
    res = np.array(res[:-1])
    # print(res[-5:])
    return res
