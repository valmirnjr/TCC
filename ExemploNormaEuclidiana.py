# Euclidian norm with NumPy
import numpy as np

x_cpu = np.array([1, 2, 3])
norma = np.linalg.norm(x_cpu)

# Euclidian norm with CuPy
import cupy as cp

x_gpu = cp.array([1, 2, 3])
norma = cp.linalg.norm(x_gpu)

# Transferring array from CPU memory to GPU memory
x_gpu = cp.asarray(x_cpu)
norma = cp.linalg.norm(x_gpu)

# Customized kernel with CuPy
kernel_norma = cp.ReductionKernel(
    'float32 x',  # Parametros de entrada
    'float32 y',  # Parametros de saida
    'x * x',  # Operacao aplicada a cada elemento
    'a + b',  # Operacao de reducao
    'y = sqrt(a)',  # Operacao final
    '0',  # Valor identidade
    'norma'  # Nome do kernel
)
