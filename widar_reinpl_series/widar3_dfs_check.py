import io
import zipfile
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from widar_reinpl_series.re_inpl_widar3 import dfs

dfs_path = Path("/media/yk/D4FA828F299D817A/DFS.zip")
csi_path = Path("/media/yk/linux_data/csi_dataset_survey/dataset/widar/mat_save_folder")
archive = zipfile.ZipFile(dfs_path, "r")
for n0 in sorted(csi_path.glob("*.mat")):
    print(n0)
    time_name = n0.name.split("_")[0]
    file_name = "user" + n0.stem.split("user")[-1][:-2] + time_name + ".mat"
    user_name = n0.stem.split("_")[-1].split("-")[0]
    ori_csi = loadmat(n0)["csi_data"].reshape(-1, 90)
    print(ori_csi.shape)
    rx_idx = int(n0.stem.split("r")[-1])
    dfs_name = "DFS/" + time_name + "/" + user_name + "/" + file_name
    ori_dfs = loadmat(io.BytesIO(archive.read(dfs_name)))["doppler_spectrum"][
        rx_idx - 1
    ]
    re_inpl_dfs = dfs(ori_csi)
    print(ori_dfs.shape, re_inpl_dfs.shape)
    fig, ax = plt.subplots(1, 2)
    ax[0].contour(ori_dfs)
    ax[1].contour(re_inpl_dfs)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # img is rgb, convert to opencv's default bgr
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    # display image with opecv or any operation you like
    cv.imshow("plot", img)
    cv.waitKey(500)
