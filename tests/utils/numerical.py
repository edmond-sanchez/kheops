import numpy as np
import pytest
from numpy.testing import assert_allclose

from kheops.utils.numerical import numerical_derivatives, numerical_gradient_hessian


def test_numerical_method_on_polynomial():
    z = np.array([-2.0, 0.0, 3.0])
    first, second = numerical_derivatives(lambda x: x**4, z, h=0.5)
    assert_allclose(first, 4*z**3, atol=1e-12)
    assert_allclose(second, 12*z**2, atol=1e-12)
    first, second = numerical_derivatives(lambda x: x**4, 2.0)
    assert_allclose([first, second], [32.0, 48.0], atol=1e-12)


@pytest.mark.parametrize("h", [0, -1, np.nan, np.inf])
def test_invalid_step(h):
    with pytest.raises(ValueError):
        numerical_derivatives(lambda x: x**2, 1.0, h)


def test_numerical_helper_on_polynomial():
    # f = z² + 3zp + 2p²: checks the mixed derivative as well as the diagonals.
    def f(x):
        z, p = x[..., 0], x[..., 1]
        return z**2 + 3*z*p + 2*p**2

    x = np.array([2.0, -1.0])
    gradient, hessian = numerical_gradient_hessian(f, x, steps=(0.1, 0.1))
    assert_allclose(gradient, [1.0, 2.0], atol=1e-12)
    assert_allclose(hessian, [[2.0, 3.0], [3.0, 4.0]], atol=1e-12)
