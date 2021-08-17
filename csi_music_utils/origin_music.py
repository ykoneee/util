import numpy as np

from gui import GUI

grid_plot_theta_vals = np.linspace(0, 180, 200) * np.pi / 180

# fc = 5785e6
# fs = 20e6
c = 3e8


# d = c / fc / 2


class SignalSimulator:
    def __init__(self):
        self.N = 5
        self.M = 30
        self.p = 40
        self.freq = 5785e6
        self.d = 0.025
        self.doa = np.array([15, 50, 85, 110, 145])

    def gen_signal(self):
        t_val = np.linspace(1 / 20e6, 1, self.p)
        cov = np.diag(np.ones(self.N))
        # cov = np.random.normal(size=(self.N, self.N))
        # cov = cov.T @ cov
        amp = np.random.multivariate_normal(
            mean=np.zeros(self.N), cov=cov, size=self.p
        ).T
        s = amp * np.exp(1j * 2 * np.pi * self.freq * t_val)
        A = SteeringVector(self.doa[: self.N] * np.pi / 180, self.M, self.freq, self.d)
        noise = np.random.multivariate_normal(
            mean=np.zeros(self.M), cov=np.diag(np.ones(self.M)), size=self.p
        ).T
        X = A @ s + noise * 0.2
        return X


def SteeringVector(theta, M, f, d):
    return np.exp(np.outer(np.arange(M), (-1j * 2 * np.pi * f * d * np.cos(theta) / c)))


def P_MU(theta, Un, M, fc, d):
    t2 = SteeringVector(theta, M, fc, d).conj().T @ Un
    return 1 / np.einsum("ij,ij->i", t2, t2.conj(), optimize=True)


sim = SignalSimulator()
gui = GUI(sim)
gui.show()
for i in range(10000):
    if i % 100 == 0:
        print(i)

    X = sim.gen_signal()
    # finding covariance matrix of X
    # idx = np.sort(np.random.permutation(X.shape[1]))[:20]
    # S = X[:, idx] @ X[:, idx].conj().T / sim.p
    S = X @ X.conj().T / sim.p

    # finding eigen values and eigen vectors
    eigvals, eigvecs = np.linalg.eig(S)
    eigvals = eigvals.real

    # finding norm of eigvals so that they can be sorted in decreasing order
    eignorms = np.abs(eigvals)

    # sorting eig vals and eig vecs in decreasing order of eig vals

    idx = eignorms.argsort()[::-1]
    eignorms = eignorms[idx]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # separating source and noise eigvectors
    Us, Un = eigvecs[:, : sim.N], eigvecs[:, sim.N :]

    P_MU_vals = P_MU(grid_plot_theta_vals, Un, sim.M, sim.freq, sim.d).real

    # P_MU_vals = np.array([P_MU2(val) for val in theta_vals]).real

    # print("doas=", doa * 180 / np.pi)
    # print("amps=", amp)

    # peak_indices = scipy.signal.find_peaks_cwt(vector=P_MU_vals, widths = 5*np.ones(len(P_MU_vals)), gap_thresh=None, min_length=None, min_snr=3, noise_perc=1)

    gui.update_music(P_MU_vals)
gui.exit()
