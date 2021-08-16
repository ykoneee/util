from dataset_manager import CSI_ZARR_DATASET, ZARR_DATASET_CREATER
from widar_reinpl_series.re_inpl_filter import apply_csi_filter

# print("in music_process_utils.py")
import concurrent.futures
import io
import itertools
import re
import zipfile
from pathlib import Path

import numpy as np
import tqdm
from sklearn.cluster import DBSCAN

from . import NDMUSIC, CSIBaseModel
from math_utils import split_array_bychunk, minmax_scale, local_maximum
from widar_reinpl_series import gen_fir_filter


from time import time


def write_npy_in_zip(f_name, array, zip_f):
    with io.BytesIO() as f:
        np.save(f, array, allow_pickle=False)
        f.flush()
        f.seek(0)
        data = f.read()
        # data = blosc.compress(data, typesize=4)
        zip_f.writestr(f_name, data=data)
    # data = blosc.pack_array(array)
    # zip_f.writestr(f_name, data=data)


# raw 4.3g
# defalut bz 3.8g
# default 4 size 3.7g
# array pack 3.8g


def load_npy_from_bytes(bytes_data):
    return np.load(io.BytesIO(bytes_data))


def gen_music_from_widar3(raw_path):
    w = (50, 2, 24)
    sspace_grid = (30, 30, 30)
    tof_sspace_max = 0.5
    dfs_sspace_max = 25
    aoa_sspace_max = 180
    model = CSIBaseModel(
        sample_len=100, channel=157, sample_freq=1000, subcarrier_num=30
    )
    estimater = NDMUSIC(
        model,
        window=w,
        sspace_grid=sspace_grid,
        tof_sspace_max=tof_sspace_max,
        dfs_sspace_max=dfs_sspace_max,
        aoa_sspace_max=aoa_sspace_max,
        gpu_data_slice_window=(10, 10, 10),
    )
    denoise = False
    estimater.check_smooth_shape()

    filter_fir = gen_fir_filter(1000, irlen=250)
    music_config_str = f"{w}_{sspace_grid}_{tof_sspace_max}_{dfs_sspace_max}_{aoa_sspace_max}_{'denoise' if denoise else ''}".replace(
        " ", ""
    )
    print(music_config_str)

    path = Path(raw_path)
    out_path = path.parent / f"music_{music_config_str}"
    d = CSI_ZARR_DATASET(path)
    d_creater = ZARR_DATASET_CREATER(out_path, "data", 4 * 18)

    with tqdm.tqdm(total=sum([x["csi_data_raw"].shape[0] for x in d]) // 100) as pbar:
        for i in range(len(d)):
            raw = d[i]["csi_data_raw"]
            # raw = raw[..., resample_subcarrier_idx_114_30]
            if denoise:
                raw = apply_csi_filter(raw, filter_fir)
                # raw, _ = conj_multi_csi(raw)

            raw, _ = split_array_bychunk(raw, 100, cat_residual=False)

            result = []
            for sample_windowd in raw:
                X = sample_windowd
                X = estimater.signal_3D_smooth(X)
                P = estimater.estimate(X)

                # P = minmax_scale(P)
                result.append(P)
                pbar.update(1)
            result = np.asarray(result)
            d_creater.append(result)
    d_creater.close()


def load_npy_from_zip():
    path = Path("subset1_np/music_(25, 1, 7)_(40, 40, 40)_1_25_180.zip")
    array = []
    pattern = re.compile(r"_(\d+).npy")
    with zipfile.ZipFile(path, "r") as zip_file:
        for i in sorted(
            zip_file.infolist(),
            key=lambda x: int(re.findall(pattern, x.filename)[0]),
        ):
            f = io.BytesIO(zip_file.read(i))
            array.append(np.load(f))
    print(len(array))
    return array


def _load_npy(zipf_path, zipinfo_list):
    array = []
    with zipfile.ZipFile(zipf_path, "r") as zip_file:
        for i in zipinfo_list:
            array.append(load_npy_from_bytes(zip_file.read(i)))
    return array


def load_npy_from_zip_parallel(path):
    path = Path(path)
    assert path.exists()
    array = None
    pattern = re.compile(r"_(\d+).npy")
    with zipfile.ZipFile(path, "r") as zip_file:
        infolist = split_array_bychunk(
            sorted(
                zip_file.infolist(),
                key=lambda x: int(re.findall(pattern, x.filename)[0]),
            ),
            10,
        )
    with concurrent.futures.ProcessPoolExecutor(6) as executor:
        array = list(
            itertools.chain.from_iterable(
                executor.map(_load_npy, itertools.repeat(path), infolist)
            )
        )

    return array


def load_npy_from_zip_parallel_test():
    t1 = time()
    d1 = load_npy_from_zip_parallel()
    print("parallel finished")
    t2 = time()
    d2 = load_npy_from_zip()
    t3 = time()
    print(t3 - t2, t2 - t1)


def music_clustering_maximum(P, clip_len, verbose=False):
    window_shape = np.asarray(P.shape)
    P = minmax_scale(P)

    P[P < 0.07] = 0
    coords, values = local_maximum(P, local_size=1)

    coords = coords[values != 0]
    values = values[values != 0]

    params = coords / window_shape[None, :]
    clustering = DBSCAN(eps=0.05, min_samples=1).fit(params)
    # clustering = DBSCAN(eps=4, min_samples=1).fit(params)

    cluster_len = clustering.labels_.max() + 1
    cluster_core_mean_array = []
    cluster_value_mean_array = []
    for i in range(cluster_len):
        cluster_core_mean_array.append(np.mean(params[clustering.labels_ == i], axis=0))
        cluster_value_mean_array.append(
            np.mean(values[clustering.labels_ == i], axis=0)
        )
    cluster_core_mean_array = np.asarray(cluster_core_mean_array)
    cluster_value_mean_array = np.asarray(cluster_value_mean_array)
    cluster_core_mean_array, cluster_value_mean_array = cluster_len_clip(
        cluster_core_mean_array, cluster_value_mean_array, clip_len
    )
    if verbose:
        print(params)
        print(clustering.core_sample_indices_[-1])
        print(clustering.labels_[-1])
        print(cluster_core_mean_array)
        print(cluster_value_mean_array)
    return (params, values, cluster_core_mean_array, cluster_value_mean_array)


def cluster_len_clip(p, v, max_len):
    if len(v) > max_len:
        _idx = np.argsort(v)
        v = v[_idx][::-1]
        p = p[_idx][::-1]
        v = v[:max_len]
        p = p[:max_len]

    return (p, v)


def music_list_localmaximum(music_list, clip_len, process_idx):
    music_localmax = []
    for m in music_list:
        temp = []
        for mm in m:
            p1, v1, p2, v2 = music_clustering_maximum(mm, clip_len)
            temp.append(np.concatenate([p2, v2[:, None]], axis=1))
        music_localmax.append(temp)
    return music_localmax, process_idx


def music_list_localmaximum_parallel(music_list, clip_len):
    #
    # print("localmax_process....")
    music_list_splited = split_array_bychunk(music_list, 10)
    # print(list(map(len, music_list_splited)))

    with concurrent.futures.ProcessPoolExecutor(4) as executor:
        ret, idx = zip(
            *executor.map(
                music_list_localmaximum,
                music_list_splited,
                itertools.repeat(clip_len),
                range(len(music_list_splited)),
            )
        )
    # print("localmax_process finish")
    return list(itertools.chain.from_iterable(ret))


def cluster_test(data, ax):
    import seaborn

    l = sum(map(len, data))
    v2_lens = []
    v2_values = []
    v1_lens = []
    v1_values = []
    l3 = []
    for d in data:
        print(l)
        l -= len(d)
        for dd in d:
            p1, v1, p2, v2 = music_clustering_maximum(dd, clip_len=24)
            l3.append(len(v1) - len(v2))

            v2_lens.append(v2.shape[0])
            v2_values.append(v2)
            v1_lens.append(v1.shape[0])
            v1_values.append(v1)

    l3 = np.array(l3)
    l3 = l3[l3 != 0]
    print(
        "cluster diff:",
        l3.sum(),
        l3.shape,
    )
    print(f"lmax1 len: max {max(v1_lens)} min {min(v1_lens)} sum {sum(v1_lens)}")
    print(f"lmax2 len: max {max(v2_lens)} min {min(v2_lens)} sum {sum(v2_lens)}")

    seaborn.histplot(v2_lens, binwidth=1, ax=ax[0, 0])
    seaborn.histplot(np.concatenate(v2_values), ax=ax[0, 1])
    seaborn.histplot(v1_lens, binwidth=1, ax=ax[1, 0])
    seaborn.histplot(np.concatenate(v1_values), ax=ax[1, 1])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 4)
    data1 = np.load("subset1_np/music.npy", allow_pickle=True)
    data2 = load_npy_from_zip_parallel()
    cluster_test(data1, ax[:, 0:2])
    cluster_test(data2, ax[:, 2:4])
    plt.show()
    # gen_music_lmax_parallel_test()
    # load_npy_from_zip_parallel_test()

    pass
