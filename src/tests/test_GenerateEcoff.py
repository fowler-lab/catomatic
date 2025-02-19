import pytest
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult
from catomatic.Ecoff import EcoffGenerator


@pytest.fixture
def wt_samples():
    """Fixture for sample data (wild-type, no mutations)."""
    samples = pd.DataFrame({"UNIQUEID": ["A", "B", "C"], "MIC": ["1", "2", "3"]})
    mutations = pd.DataFrame({"UNIQUEID": ["A"], "MUTATION": [None]})
    return samples, mutations


@pytest.fixture
def mixed_variants():
    """Fixture for data with both wild-type and mutant samples."""
    samples = pd.DataFrame(
        {"UNIQUEID": ["A", "B", "C", "D"], "MIC": ["1", "<=2", ">3", "4"]}
    )
    mutations = pd.DataFrame(
        {
            "UNIQUEID": ["A", "B", "C"],
            "MUTATION": [None, "mut1@G12G", "mut2@A13V"],
        }
    )
    return samples, mutations


def test_flag_mutants(wt_samples, mixed_variants):
    """Test mutant flagging with simple sample data (no mutations)."""
    samples, mutations = wt_samples
    ecoff = EcoffGenerator(samples, mutations)
    assert all(
        ecoff.df["MUTANT"] == False
    ), "All samples should be flagged as non-mutants."

    samples, mutations = mixed_variants
    ecoff = EcoffGenerator(samples, mutations)
    expected_mutant_flags = [
        False,
        False,
        True,
        False,
    ]  # Based on mutation column values
    assert (
        list(ecoff.df["MUTANT"]) == expected_mutant_flags
    ), f"Expected mutant flags {expected_mutant_flags}, got {list(ecoff.df['MUTANT'])}"


def test_define_intervals(wt_samples, mixed_variants):
    """Test interval definition for uncensored data."""
    samples, mutations = wt_samples
    ecoff = EcoffGenerator(samples, mutations)
    y_low, y_high = ecoff.define_intervals(ecoff.df[ecoff.df["MUTANT"] == False])

    expected_y_low = [-1, 0, 0.58]
    expected_y_high = [0, 1, 1.58]

    assert np.allclose(
        y_low, expected_y_low, atol=1e-2
    ), f"Expected y_low {expected_y_low}, got {y_low}"
    assert np.allclose(
        y_high, expected_y_high, atol=1e-2
    ), f"Expected y_high {expected_y_high}, got {y_high}"

    samples, mutations = mixed_variants
    ecoff = EcoffGenerator(samples, mutations, censored=True)
    y_low, y_high = ecoff.define_intervals(ecoff.df)

    expected_y_low = [-1, -19.93, 1.58, 1.0]
    expected_y_high = [0, 1, np.inf, 2.0]

    assert np.allclose(
        y_low, expected_y_low, atol=1e-2
    ), f"Expected y_low {expected_y_low}, got {y_low}"
    assert np.allclose(
        y_high, expected_y_high, atol=1e-2
    ), f"Expected y_high {expected_y_high}, got {y_high}"


def test_log_transf_intervals(wt_samples):
    """Test log transformation of intervals with default dilution factor (2)."""
    samples, mutations = wt_samples
    ecoff = EcoffGenerator(samples, mutations)
    y_low, y_high = np.array([0.5, 1.0, 1.5]), np.array([1.0, 2.0, 3.0])
    y_low_log, y_high_log = ecoff.log_transf_intervals(y_low, y_high)

    # Expected values in log2 space
    expected_y_low_log = [-1, 0, 0.585]
    expected_y_high_log = [0, 1, 1.585]
    assert np.allclose(
        y_low_log, expected_y_low_log, atol=1e-2
    ), f"Expected y_low_log {expected_y_low_log}, got {y_low_log}"
    assert np.allclose(
        y_high_log, expected_y_high_log, atol=1e-2
    ), f"Expected y_high_log {expected_y_high_log}, got {y_high_log}"


def test_fit_model(wt_samples):
    """Test model fitting with simple sample data."""
    samples, mutations = wt_samples
    ecoff = EcoffGenerator(samples, mutations)
    result = ecoff.fit()

    # Validate the optimization result
    assert isinstance(result, OptimizeResult), "Expected an OptimizeResult from fit."
    assert np.isfinite(result.x[0]), f"Expected finite mu, got {result.x[0]}"
    assert np.isfinite(result.x[1]), f"Expected finite log(sigma), got {result.x[1]}"


def test_generate_ecoff(wt_samples):
    """Test ECOFF generation with simple sample data."""
    samples, mutations = wt_samples
    ecoff = EcoffGenerator(samples, mutations)
    ecoff_value, z_99, mu, sigma, model = ecoff.generate()

    assert isinstance(ecoff_value, float), "ECOFF should be a float."
    assert isinstance(
        model, OptimizeResult
    ), "Expected an OptimizeResult for the model."

    assert np.allclose(ecoff_value, 3.25, atol=1e-2), f"ECOFF should be around 3.28"
    assert z_99 > mu, "99th percentile (z_99) should be greater than mean (mu)."
    assert sigma > 0, f"Sigma should be positive, got {sigma}"
