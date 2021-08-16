import cupy
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch
from opt_einsum import contract

a_shape = (20, 20, 20, 1, 1500)
b_shape = (1500, 1496)
c_shape = (1496, 1500)
d_shape = (20, 20, 20, 1500, 1)

from time import time


@jax.jit
def jax_f(a, b, c, d):
    return jnp.matmul(jnp.matmul(jnp.matmul(a, b), c), d)


def test_jax(i):
    print(i)
    a = np.random.rand(*a_shape) + 1j * np.random.rand(*a_shape)
    b = np.random.rand(*b_shape) + 1j * np.random.rand(*b_shape)
    c = np.random.rand(*c_shape) + 1j * np.random.rand(*c_shape)
    d = np.random.rand(*d_shape) + 1j * np.random.rand(*d_shape)

    t11 = time()
    p = jnp.matmul(jnp.matmul(jnp.matmul(a, b), c), d)
    t12 = time()
    # print(p.shape)
    return t12 - t11


def test_jax_jit(i):
    print(i)
    a = np.random.rand(*a_shape) + 1j * np.random.rand(*a_shape)
    b = np.random.rand(*b_shape) + 1j * np.random.rand(*b_shape)
    c = np.random.rand(*c_shape) + 1j * np.random.rand(*c_shape)
    d = np.random.rand(*d_shape) + 1j * np.random.rand(*d_shape)

    t11 = time()
    p = jax_f(a, b, c, d)
    t12 = time()
    # print(p.shape)
    return t12 - t11


def test_np(i):
    print(i)
    a = np.random.rand(*a_shape) + 1j * np.random.rand(*a_shape)
    b = np.random.rand(*b_shape) + 1j * np.random.rand(*b_shape)
    c = np.random.rand(*c_shape) + 1j * np.random.rand(*c_shape)
    d = np.random.rand(*d_shape) + 1j * np.random.rand(*d_shape)

    t11 = time()
    p = np.matmul(np.matmul(np.matmul(a, b), c), d)
    t12 = time()
    # print(p.shape)
    return t12 - t11


def test_oe(i):
    print(i)
    a = np.random.rand(*a_shape) + 1j * np.random.rand(*a_shape)
    b = np.random.rand(*b_shape) + 1j * np.random.rand(*b_shape)
    c = np.random.rand(*c_shape) + 1j * np.random.rand(*c_shape)
    d = np.random.rand(*d_shape) + 1j * np.random.rand(*d_shape)
    t11 = time()
    contract("...ij,jk,kl,...lm", a, b, c, d, backend="jax")
    t12 = time()
    return t12 - t11


def test_tf(i):
    print(i)

    a = np.random.rand(*a_shape) + 1j * np.random.rand(*a_shape)
    b = np.random.rand(*b_shape) + 1j * np.random.rand(*b_shape)
    c = np.random.rand(*c_shape) + 1j * np.random.rand(*c_shape)
    d = np.random.rand(*d_shape) + 1j * np.random.rand(*d_shape)
    a = tf.convert_to_tensor(a)
    b = tf.convert_to_tensor(b)
    c = tf.convert_to_tensor(c)
    d = tf.convert_to_tensor(d)
    t11 = time()
    tf.matmul(tf.matmul(tf.matmul(a, b), c), d)
    t12 = time()
    return t12 - t11


def test_pytorch_ein(i):
    print(i)
    a = torch.randn(*a_shape, dtype=torch.complex128, device="cuda")
    b = torch.randn(*b_shape, dtype=torch.complex128, device="cuda")
    c = torch.randn(*c_shape, dtype=torch.complex128, device="cuda")
    d = torch.randn(*d_shape, dtype=torch.complex128, device="cuda")
    t11 = time()
    contract("ij...,jk,kl,lm...", a, b, c, d, backend="torch").to("cpu")
    t12 = time()
    return t12 - t11


def test_pytorch(i):
    print(i)
    a = torch.randn(*a_shape, dtype=torch.complex64, device="cuda")
    b = torch.randn(*b_shape, dtype=torch.complex64, device="cuda")
    c = torch.randn(*c_shape, dtype=torch.complex64, device="cuda")
    d = torch.randn(*d_shape, dtype=torch.complex64, device="cuda")

    # a = a.view(-1, *a_shape[3:])
    # d = d.view(-1, *d_shape[3:])
    t11 = time()
    r = torch.matmul(torch.matmul(torch.matmul(a, b), c), d).to("cpu")

    t12 = time()
    return t12 - t11


def test_cupy(i):
    print(i)
    a = np.random.rand(*a_shape) + 1j * np.random.rand(*a_shape)
    b = np.random.rand(*b_shape) + 1j * np.random.rand(*b_shape)
    c = np.random.rand(*c_shape) + 1j * np.random.rand(*c_shape)
    d = np.random.rand(*d_shape) + 1j * np.random.rand(*d_shape)
    t11 = time()
    p = cupy.matmul(cupy.matmul(cupy.matmul(a, b), c), d).get()
    print(p.shape)
    t12 = time()
    return t12 - t11


n = 25
# b = [test_jax(i) for i in range(n)]
b = [test_jax_jit(i) for i in range(n)]
# b = [test_oe(i) for i in range(n)]
# b = [test_np(i) for i in range(n)]
# b = [test_pytorch(i) for i in range(n)]
# b = [test_pytorch_ein(i) for i in range(n)]
# b = [test_tf(i) for i in range(n)]
# b = [test_cupy(i) for i in range(n)]
print(np.mean(b))
