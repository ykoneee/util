from csi_music_utils.music_process_utils import cluster_test
from dataset_manager import CSI_ZARR_DATASET

rawpath1 = "dataset/subset1_np_{'receiverid': (3,), 'face_orientation': (1, 2, 3), 'location': (5,)}/music_(35,2,23)_(30,30,30)_0.5_25_180_"
rawpath2 = "dataset/subset1_np2/music_(50,2,23)_(30,30,30)_0.5_25_180_"

music_list1 = CSI_ZARR_DATASET(rawpath1).get_all_data()["data"]
music_list2 = CSI_ZARR_DATASET(rawpath2).get_all_data()["data"]
# music_localmax = np.array(
#     music_list_localmaximum_parallel(music_list, 16), dtype=object, copy=False
# )


import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 4)

cluster_test(music_list1, ax[:, 0:2])
cluster_test(music_list2, ax[:, 2:4])
plt.show()
