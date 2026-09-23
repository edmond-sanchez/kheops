import numpy as np


def numerical_derivatives(f, z, h=0.5):
    if np.ndim(h) != 0 or not np.isfinite(h) or h <= 0:
        raise ValueError("h must be a positive finite scalar.")
    z = np.asarray(z, dtype=float)
    fm2, fm1 = f(z - 2*h), f(z - h)
    f0 = f(z)
    fp1, fp2 = f(z + h), f(z + 2*h)
    first = (fm2 - 8*fm1 + 8*fp1 - fp2) / (12*h)
    second = (-fp2 + 16*fp1 - 30*f0 + 16*fm1 - fm2) / (12*h*h)
    return first, second
