"""Test for snapshot positioning algorithm.

"""

__authors__ = "Bradley Collicott"
__date__ = "22 October 2021"

import pytest
import numpy as np

from gnss_lib_py.algorithms import snapshot

# Defining test fixtures
@pytest.fixture(name="tolerance")
def fixture_tolerance():
    """Decimal threshold for test pass/fail criterion."""
    return 7

@pytest.fixture(name="set_user_states")
def fixture_set_user_states():
    """ Set the location and clock bias of the user receiver in Earth-Centered,
    Earth-Fixed coordinates.

    Returns
    -------
    x_u : float
        User x position, scalar, units [m]
    y_u : float
        User y position, scalar, units [m]
    z_u : float
        User z position, scalar, units [m]
    b_clk_u : float
        Range bias due to user clock offset (c*dt), scalar, units [m]
    """
    x_u = 3678300.0
    y_u = 3678300.0
    z_u = 3678300.0
    b_clk_u = 10.0
    return x_u, y_u, z_u, b_clk_u

@pytest.fixture(name="set_sv_states")
def fixture_set_sv_states():
    """Set the location and clock bias of 4 satellite in Earth-Centered,
    Earth-Fixed coordinates.

    Returns
    -------
    x_sv : np.ndarray
        Satellite x positions, 4-by-3, units [m]
    y_sv : np.ndarray
        Satellite y positions, 4-by-3, units [m]
    z_sv : np.ndarray
        Satellite z positions, 4-by-3, units [m]
    b_clk_u : np.ndarray
        Range biases due to satellite clock offset (c*dt), 4-by-3, units [m]

    References
    ----------
    .. [1] Weiss, M., & Ashby, N. (1999).
       Global Positioning System Receivers and Relativity.
    """
    x_sv = np.array([13005878.255, 20451225.952, 20983704.633, 13798849.321])
    y_sv = np.array([18996947.213, 16359086.310, 15906974.416, -8709113.822])
    z_sv = np.array([13246718.721, -4436309.875, 3486495.546, 20959777.407])
    b_clk_sv = np.array([5.0, 5.0, 5.0, 5.0])
    return x_sv, y_sv, z_sv, b_clk_sv

# Defining tests
def test_snapshot(set_user_states, set_sv_states, tolerance):
    """Test snapshot positioning against truth user states.

    Parameters
    ----------
    set_user_states : fixture
        Truth values for user position and clock bias
    set_sv_states : fixture
        Satellite position and clock biases
    tolerance : fixture
        Error threshold for test pass/fail
    """
    x_u , y_u , z_u , b_clk_u  = set_user_states
    x_sv, y_sv, z_sv, b_clk_sv = set_sv_states

    # Compute noise-free pseudorange measurements
    prange_measured = (np.sqrt((x_u - x_sv)**2
                             + (y_u - y_sv)**2
                             + (z_u - z_sv)**2)
                             + (b_clk_u - b_clk_sv))

    user_fix = snapshot.solvepos(prange_measured, x_sv , y_sv , z_sv , b_clk_sv)
    truth_fix = np.array([x_u, y_u, z_u, b_clk_u]).reshape([-1,1])

    np.testing.assert_array_almost_equal(user_fix, truth_fix, decimal=tolerance)
