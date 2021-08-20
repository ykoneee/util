import numpy as np
from scipy import ndimage
import scipy.interpolate


def complex_to_amp_pha(x, unwrap_axis=0):
    amp = np.abs(x)
    pha = np.unwrap(np.angle(x), axis=unwrap_axis)
    # pha = np.angle(x)
    return amp.astype(np.float32), pha.astype(np.float32)


def minmax_scale(x, min_v=0, max_v=1):
    return (x - x.min()) / (x.max() - x.min())


def local_maximum(X, local_size=1):
    size = 1 + 2 * local_size
    # footprint = np.ones((size, size, size))
    # footprint[local_size, local_size, local_size] = 0
    # X_filter = ndimage.maximum_filter(X, footprint=footprint, mode="constant")
    # X_filter = ndimage.maximum_filter(X, footprint=footprint)
    X_filter = ndimage.maximum_filter(X, size=size, mode="constant")
    mask_local_maxima = X == X_filter

    coords = np.asarray(np.where(mask_local_maxima)).T
    values = X[mask_local_maxima]

    # local_maxima_idx = np.argsort(values)
    # values = values[local_maxima_idx][::-1]
    # coords = coords[local_maxima_idx][::-1]
    return coords, values


def array_debug_infor(array):
    print(array.min(), array.max(), array.mean())


def split_array_bychunk(arr, chunksize, cat_residual=True):
    len_ = len(arr) // chunksize * chunksize
    arr, array_res = arr[:len_], arr[len_:]
    arr = [arr[i * chunksize : (i + 1) * chunksize] for i in range(len(arr) // chunksize)]
    if cat_residual:
        if len(array_res) == 0:
            return arr
        else:
            return arr + [
                array_res,
            ]
    else:
        if len(array_res) == 0:
            return arr, None
        else:
            return arr, array_res


def csi_split(data, chunk_size=100):
    result = []
    for x in data:
        x_chunked, _ = split_array_bychunk(x, chunk_size, cat_residual=False)
        x_chunked += [
            x[-chunk_size:],
        ]
        result.append(np.array(x_chunked))
    return result


def randcn(shape):
    """Samples from complex circularly-symmetric normal distribution.
    Args:
        shape (tuple): Shape of the output.

    Returns:
        ~numpy.ndarray: A complex :class:`~numpy.ndarray` containing the
        samples.
    """
    x = 1j * np.random.randn(*shape)
    x += np.random.randn(*shape)
    x *= np.sqrt(0.5)
    return x


def interpolation_1D(x, point_n, axis=0, kind="slinear"):
    if x.shape[axis] == point_n:
        return x
    f = scipy.interpolate.interp1d(
        np.linspace(0, 1024, x.shape[axis]),
        x,
        axis=axis,
        copy=False,
        kind=kind,
        assume_sorted=True,
    )
    return f(np.linspace(0, 1024, point_n))


if __name__ == "__main__":
    X = np.random.randint(0, 100, (7, 7))
    local_size = 1
    size = 1 + 2 * local_size
    footprint = np.ones((size, size))
    footprint[local_size, local_size] = 0

    X_filter = ndimage.maximum_filter(X, footprint=footprint)
    mask_local_maxima = X >= X_filter
    X_filter2 = ndimage.maximum_filter(X, size=3)
    mask_local_maxima2 = X >= X_filter2
    pass
