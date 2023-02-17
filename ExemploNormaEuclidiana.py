# Euclidian norm with NumPy
import numpy as np

x_cpu = np.array([1, 2, 3])
norma = np.linalg.norm(x_cpu)

# Euclidian norm with CuPy
import cupy as cp

x_gpu = cp.array([1, 2, 3])
norma = cp.linalg.norm(x_gpu)
