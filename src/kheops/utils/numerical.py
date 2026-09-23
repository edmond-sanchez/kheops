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


def numerical_gradient_hessian(f, x, steps):
    """Gradient and Hessian of a scalar f(z, p), with separate steps for z and p.

    x has shape (..., 2). f accepts that array and returns shape (...,).
    Diagonal derivatives use the five-point formula above; the mixed derivative
    uses a four-corner central difference. Stay inside a smooth part of f.
    """
    x = np.asarray(x, dtype=float)
    hz, hp = steps
    gradient = np.empty_like(x)
    hessian = np.empty(x.shape[:-1] + (2, 2))

    for i, h in enumerate(steps):
        direction = np.eye(2)[i]
        first, second = numerical_derivatives(
            lambda offset: f(x + offset*direction), 0.0, h
        )
        gradient[..., i] = first
        hessian[..., i, i] = second

    dz = np.array([hz, 0.0])
    dp = np.array([0.0, hp])
    mixed = (f(x + dz + dp) - f(x + dz - dp)
             - f(x - dz + dp) + f(x - dz - dp)) / (4*hz*hp)
    hessian[..., 0, 1] = hessian[..., 1, 0] = mixed
    return gradient, hessian
