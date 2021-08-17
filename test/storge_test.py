import numpy as np
import zarr


def np_objseq2array():
    base = np.load(
        "/home/yk/2020_project/csi_signal_git_proj/subset1_np/music.npy",
        allow_pickle=True,
    )

    var_len = np.asarray([len(x) for x in base])
    max_len = var_len.max()
    base_len = len(base)
    element_shape = base[0].shape[1:]
    arr = np.zeros((base_len, max_len) + element_shape, dtype=base[0].dtype)

    for i in range(len(arr)):
        arr[i][: var_len[i]] = base[i]

    np.save("test", arr)


# np_objseq2array()


def zarr_load_stest():
    array = zarr.open("example.zip", mode="r")
    print(array.shape)
    for i in range(100):
        print(i)
        idx = np.random.randint(0, len(array))
        temp = array[idx]
        temp2 = temp + 1


def np_load_stest():
    array = np.load("test.npy")
    print(array.shape)
    for i in range(100):
        print(i)
        idx = np.random.randint(0, len(array))
        temp = array[idx]
        temp2 = temp + 1


# zarr_load_stest()
# np_load_stest()
pass
