import functools
import time

import numpy as np
import scipy.signal

from util.math_utils import complex_to_amp_pha


def filter_for_complex(x, f):
    return f(x=x.real) + 1j * f(x=x.imag)


def apply_csi_filter(x, filter):
    amp, pha = complex_to_amp_pha(x)
    x = filter(x=amp) * np.exp(1j * filter(x=pha))
    return x


def np_contiguous(f):
    return lambda *a, **ka: np.ascontiguousarray(f(*a, **ka))


def gen_fir_filter(sample_rate, irlen=None):
    samp_rate = sample_rate
    half_rate = samp_rate / 2
    uppe_stop = 50

    f = np.array(
        [
            0,
            uppe_stop / half_rate,
            int(uppe_stop * 1.1) / half_rate,
            1,
        ]
    )
    m = np.array([1, 1, 0, 0])
    b = scipy.signal.firls(27, f, m)
    a = 1

    ret = functools.partial(scipy.signal.filtfilt, b=b, a=a, axis=0, method="gust", irlen=irlen)
    return np_contiguous(ret)


def gen_iir_filter(sample_rate, c_freq, c_order, mode, output="sos", bi_filter=True):
    samp_rate = sample_rate
    half_rate = samp_rate / 2

    res = scipy.signal.butter(c_order, c_freq / half_rate, mode, output=output)

    if output == "sos":
        if bi_filter:
            ret = functools.partial(scipy.signal.sosfiltfilt, sos=res, axis=0)
        else:
            ret = functools.partial(scipy.signal.sosfilt, sos=res, axis=0)
    elif output == "ba":
        if bi_filter:
            ret = functools.partial(scipy.signal.filtfilt, b=res[0], a=res[1], axis=0, method="gust")
        else:
            ret = functools.partial(scipy.signal.lfilter, b=res[0], a=res[1], axis=0)
    else:
        raise Exception
    return np_contiguous(ret)


def fir_speed_test():
    f1 = gen_fir_filter(1000)
    f2 = gen_iir_filter(1000)

    r1 = 0
    r2 = 0
    for i in range(50):
        print(i)
        x = np.random.rand(1800, 3, 114) + 1j * np.random.rand(1800, 3, 114)
        t0 = time.time()
        x1 = f1(x=x)
        t1 = time.time()
        r1 += t1 - t0
    for i in range(50):
        print(i)
        x = np.random.rand(1800, 3, 114) + 1j * np.random.rand(1800, 3, 114)
        t0 = time.time()
        x1 = f2(x=x)
        t1 = time.time()
        r2 += t1 - t0
    print(f"r1: {r1 / 50} r2: {r2 / 50}")


if __name__ == "__main__":
    fir_speed_test()
