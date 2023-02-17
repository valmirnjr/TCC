import cupy as cp
import numpy as np
from cupyx.scipy import signal


def bpsGPU(Ei, N, X, B):
    # Etapa 1
    Ei = cp.asarray(Ei)
    X = cp.asarray(X) # Simbolos de referencia da constelacao
    ϕ_test = np.arange(B) * (np.pi / 2) / B
    ϕ_test_gpu = cp.asarray(ϕ_test)
    window_filter = cp.ones((N, 1, 1)) # Filtro de convolucao S

    nModes = Ei.shape[1]
    zeroPad = cp.zeros((N // 2, nModes))
    Ei_gpu = cp.concatenate(
        (zeroPad, Ei, zeroPad)
    )  

    # Etapa 2
    Ei_rotated = Ei_gpu[:, :, cp.newaxis] * ϕ_test_gpu)[None, None, :]
    dist = cp.absolute(cp.subtract(
        Ei_rotated[:, :, :, None], X[None, None, None, :])) ** 2
    min_dist = cp.min(dist, axis=3)

    window_sums = signal.oaconvolve(min_dist, window_filter, mode="valid")

    # Etapa 3
    ind_rot = cp.argmin(window_sums, axis=2)
    θ = ϕ_test_gpu[ind_rot]

    return cp.asnumpy(θ)
