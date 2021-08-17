import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

fig, axs = plt.subplots(2)
fs = 1000  # Hz
desired = (1, 1, 0, 0)
for bi, bands in enumerate(((0, 30, 33, 500), (0, 100, 101, 500))):
    tap = 129
    fir_firls = signal.firls(tap, bands, desired, fs=fs)
    fir_remez = signal.remez(tap, bands, desired[::2], fs=fs)
    fir_firwin2 = signal.firwin2(tap, bands, desired, fs=fs)
    [lb, la] = signal.butter(9, bands[1] / 500, "lowpass")
    hs = list()
    ax = axs[bi]
    for fir in (fir_firls, fir_remez, fir_firwin2):
        freq, response = signal.freqz(fir)
        hs.append(ax.semilogy(0.5 * fs * freq / np.pi, np.abs(response))[0])

    freq, response = signal.freqz(lb, la)
    hs.append(ax.semilogy(0.5 * fs * freq / np.pi, np.abs(response))[0])
    for band, gains in zip(
        zip(bands[::2], bands[1::2]), zip(desired[::2], desired[1::2])
    ):
        ax.semilogy(band, np.maximum(gains, 1e-7), "k--", linewidth=2)
    if bi == 0:
        ax.legend(
            hs, ("firls", "remez", "firwin2", "iir"), loc="lower center", frameon=False
        )
    else:
        ax.set_xlabel("Frequency (Hz)")
    ax.grid(True)
    ax.set(title="Band-pass %d-%d Hz" % bands[2:4], ylabel="Magnitude")

fig.tight_layout()
plt.show()
