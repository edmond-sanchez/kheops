import numpy as np
import pytest
from numpy.testing import assert_allclose

from kheops.utils.numerical import numerical_derivatives


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
