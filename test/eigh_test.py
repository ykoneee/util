import os
from time import time

import cupy
import numpy as np
import scipy.linalg

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "true"
from jax.lax_linalg import eigh as jax_eigh
from math_utils import randcn

n = 25
ashape = (200, 200)


def randcn_H(shape):
    a = randcn(shape).astype(np.complex64)
    return a @ a.conj().T


def np_test():
    l = []
    for i in range(n):
        a = randcn_H(ashape)
        t1 = time()
        w, v = np.linalg.eigh(a)
        t2 = time()
        l.append(t2 - t1)
    print(np.mean(l))


def scipy_test():
    l = []
    for i in range(n):
        a = randcn_H(ashape)
        t1 = time()
        w, v = scipy.linalg.eigh(a)
        t2 = time()
        l.append(t2 - t1)
    print(np.mean(l))


def jax_test():
    l = []
    for i in range(n):
        a = randcn_H(ashape)
        t1 = time()
        w, v = jax_eigh(a)
        w = np.asarray(w)
        v = np.asarray(v)
        t2 = time()
        l.append(t2 - t1)
    print(np.mean(l))


def cupy_test():
    l = []
    for i in range(n):
        a = randcn_H(ashape)
        a = cupy.asarray(a)
        t1 = time()
        w, v = cupy.linalg.eigh(a)
        w = w.get()
        v = v.get()
        t2 = time()
        l.append(t2 - t1)
    print(np.mean(l))


def jax_cupy_eq_test():
    for i in range(1):
        a = randcn_H(ashape)

        a1 = cupy.asarray(a)
        w1, v1 = cupy.linalg.eigh(a1)
        w1 = w1.get()
        v1 = v1.get()

        v2, w2 = jax_eigh(a, symmetrize_input=False)
        w2 = np.asarray(w2)
        v2 = np.asarray(v2)

        w3, v3 = scipy.linalg.eigh(a)
        w4, v4 = np.linalg.eigh(a)

        print(np.allclose(w1, w2))
        print(np.allclose(v1, v2))

        print(np.allclose(w1, w3))
        print(np.allclose(v1, v3))

        print(np.allclose(w1, w4))
        print(np.allclose(v1, v4))
        # DIFF!!!


# np_test()
# scipy_test()
# jax_test()
# cupy_test()
jax_cupy_eq_test()
