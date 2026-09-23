import numpy as np
import pytest
from numpy.testing import assert_allclose

from kheops.environment.ssp import DownwardRefractingSSP, MunkSSP
from kheops.utils.numerical import numerical_derivatives

# Independent reference calculations: Python Decimal, 50-digit precision,
# evaluated from the published equations and their analytic derivatives.
# Columns: depth, speed, first derivative, second derivative.
DOWNWARD = np.array([
    [0, 1677.3319, -1.2286762, 0.002700083157864773],
    [66.7, 1600.934733783636, -1.068319873666596, 0.002138701837845048],
    [800, 1138.115084645785, -0.3838300430484373, 0.0003883407853936422],
    [1000, 1068.334694322834, -0.3174698035709138, 0.0002830210702178085],
    [2000, 846.0941681714865, -0.1577017566348050, 0.00008818112090093960],
])
MUNK = np.array([
    [0, 1548.521015173678, -0.1086631002671974, 0.0001933396808844458],
    [1000, 1501.381592389554, -0.009975230895172535, 0.00004151218954286899],
    [1300, 1500.0, 0.0, 0.00002616568047337278],
    [3000, 1518.666635783122, 0.01576375559993104, 0.000001913748781171185],
    [4999, 1551.893786517942, 0.01695025742801751, 8.836135334584702e-8],
    [5000, 1551.910736819529, 0.0, 0.0],
    [6000, 1551.910736819529, 0.0, 0.0],
])


@pytest.mark.parametrize("profile, table", [
    (DownwardRefractingSSP(), DOWNWARD), (MunkSSP(), MUNK),
])
def test_reference_values_scalar_and_array(profile, table):
    for column, method in enumerate((profile.c, profile.gradient, profile.curvature), 1):
        assert_allclose(method(table[:, 0]), table[:, column], rtol=2e-14, atol=1e-15)
        for row in table:
            assert_allclose(method(float(row[0])), row[column], rtol=2e-14, atol=1e-15)


@pytest.mark.parametrize("profile, table", [
    (DownwardRefractingSSP(), DOWNWARD),
    # Exclude the nonsmooth cutoff and the surface (central stencil goes negative).
    (MunkSSP(), MUNK[[1, 2, 3, 4, 6]]),
])
@pytest.mark.parametrize("h", [0.2, 0.4])
def test_numerical_derivatives_against_analytic_and_reference(profile, table, h):
    table = table[table[:, 0] > 0]
    z = table[:, 0]
    first, second = numerical_derivatives(profile.c, z, h)
    # Absolute tolerances handle zero derivatives and cancellation in f''.
    for expected in (profile.gradient(z), table[:, 2]):
        assert_allclose(first, expected, rtol=1e-8, atol=2e-11)
    for expected in (profile.curvature(z), table[:, 3]):
        assert_allclose(second, expected, rtol=1e-6, atol=5e-11)


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


def test_munk_cutoff_convention_and_constant_extension():
    p = MunkSSP()
    z = np.array([5000.0, 5001.0, 6000.0])
    assert_allclose(p.c(z), p.c(5000.0), rtol=0, atol=0)
    assert_allclose(p.gradient(z), 0, rtol=0, atol=0)
    assert_allclose(p.curvature(z), 0, rtol=0, atol=0)
    # Show the corner explicitly: nonzero slope from the left, zero from right.
    h = 0.01
    left = (p.c(5000.0) - p.c(5000.0 - h)) / h
    right = (p.c(5000.0 + h) - p.c(5000.0)) / h
    assert left == pytest.approx(0.01695034572143243, abs=1e-9)
    assert right == 0.0


@pytest.mark.parametrize("method", ["c", "gradient", "curvature"])
def test_downward_invalid_domain(method):
    with pytest.raises(ValueError):
        getattr(DownwardRefractingSSP(), method)(-1000.0)
