import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
from numba import float64, njit

# from csi_dataset import get_aril_numpy_data

data = get_aril_numpy_data()["train_data"][:, :, None, :]
print(data.shape)
# data = get_signfi_raw_data()
# data = np.transpose(data, (3, 1, 2, 0))


# d_phase = np.unwrap(np.angle(data), discont=1 * np.pi, axis=1)
d_phase = np.unwrap(data, axis=1)
# d_phase = data
# subcarrier_spacing = 312.5e3
subcarrier_spacing = 0.03125

# t = np.arange(30)[:, None]
t = np.arange(52)[:, None]


@njit(float64(float64[:], float64[:, :]))
def cost_f(args, d):  # d:(30,3)
    p = args[0]
    w = args[1]
    return np.sum((d + p * t + w) ** 2)


@njit(float64[:](float64[:], float64[:, :]))
def jac(args, d):
    p = args[0]
    w = args[1]

    dp = np.sum(2 * (d + p * t + w) * t)
    dw = np.sum(2 * (d + p * t + w))
    return np.array((dp, dw))


fig = plt.figure()
j = 669
for i in range(0, 192, 6):
    ret0 = scipy.optimize.fmin_bfgs(
        cost_f,
        x0=np.array([0.0, 0.0]),
        fprime=jac,
        args=(d_phase[j, ..., i],),
        disp=False,
    )

    # plt.subplot(5, 2, 1 + (j + i * 2))
    # plt.title(f"i:{i} j:{j}")
    # plt.plot(d_phase[j, :, 0, i], "-x", label="0")

    p = ret0[0]
    w = ret0[1]
    d2 = d_phase[j, ..., i] + p * t
    plt.plot(d2[:, 0], "-x", label="1")
# plt.legend()
# fig.tight_layout()
plt.show()
