import numpy as np

from csi_music_utils.csi_music import NDMUSIC
from csi_music_utils.csi_signal_model import CSIGeneraterModel
from csi_music_utils.music_process_utils import music_clustering_maximum

data = np.load("../subset1_np/raw.npy", allow_pickle=True)
data0 = data[111]
np.random.seed(64151)
ti = 11
data0 = data0[ti * 100 : (ti + 1) * 100]

csimodel = CSIGeneraterModel()
estimater = NDMUSIC(csimodel, window=(17 * 2, 1, 5 * 2), sspace_grid=(110, 110, 60))

X = data0
# X = csimodel.gen_signal()
X = estimater.signal_3D_smooth(X)
P = estimater.estimate(X, verbose=True)

print(f"P shape:{P.shape}")

res = music_clustering_maximum(P, verbose=True)
