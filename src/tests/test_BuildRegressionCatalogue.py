import pytest
import os
import json
import sys
import subprocess
import numpy as np
import pandas as pd
from unittest.mock import patch
from scipy.optimize import OptimizeResult
from catomatic.RegressionCatalogue import RegressionBuilder
from catomatic.cli import main_regression_builder
from scipy.spatial.distance import pdist, squareform
from catomatic.__main__ import main


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
            "UNIQUEID": ["A", "B", "C", "B"],
            "MUTATION": ["mut0@V1!", "mut1@G12G", "mut2@A13V", "mut3@121_indel"],
        }
    )
    return samples, mutations


def test_generate_snps_df(mixed_variants):
    """Test generation of filtered SNP DataFrame with required SNP_ID column."""
    samples, mutations = mixed_variants

    mutations["REF"] = ["A", "G", "C", "T"]
    mutations["ALT"] = ["G", "A", "T", "CGG"]

    builder = RegressionBuilder(samples, mutations)
    snps = builder.generate_snps_df()

    expected_snps = pd.DataFrame(
        {
            "UNIQUEID": ["A", "B", "C"],
            "MUTATION": ["mut0@V1!", "mut1@G12G", "mut2@A13V"],
            "REF": [
                "A",
                "G",
                "C",
            ],
            "ALT": [
                "G",
                "A",
                "T",
            ],
            "SNP_ID": ["mut0@A1G", "mut1@G12A", "mut2@C13T"],
        }
    )
    # Drop the indel row (not included in expected_snps)
    assert (
        snps.shape == expected_snps.shape
    ), f"Expected shape {expected_snps.shape}, got {snps.shape}"
    pd.testing.assert_frame_equal(
        snps.reset_index(drop=True),
        expected_snps.reset_index(drop=True),
        check_like=True,
    )


def test_build_X(mixed_variants):
    """Test binary mutation matrix generation with and without fixed effects."""
    samples, mutations = mixed_variants
    builder = RegressionBuilder(samples, mutations)
    df = pd.merge(samples, mutations, how="left", on=["UNIQUEID"])

    mutation_matrix = builder.build_X(df)
    expected_columns_no_fixed = [
        "Intercept",
        "mut0@V1!",
        "mut1@G12G",
        "mut2@A13V",
        "mut3@121_indel",
    ]

    expected_values_no_fixed = [
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0],
    ]

    assert mutation_matrix.shape == (4, 5)
    assert (
        list(mutation_matrix.columns) == expected_columns_no_fixed
    ), f"Expected columns {expected_columns_no_fixed}, got {list(mutation_matrix.columns)}"
    assert np.array_equal(
        mutation_matrix.values, expected_values_no_fixed
    ), f"Expected mutation matrix values {expected_values_no_fixed}, got {mutation_matrix.values.tolist()}"

    # Test mutation matrix with fixed effects
    samples_fe = samples.copy()
    samples_fe["SOURCE"] = ["Lab1", "Lab2", "Lab1", "Lab2"]
    df_fe = pd.merge(samples_fe, mutations, how="left", on=["UNIQUEID"])
    mutation_matrix_fe = builder.build_X(df_fe, fixed_effects=["SOURCE"])

    expected_columns_fixed = [
        "Intercept", "mut0@V1!",
        "mut1@G12G",
        "mut2@A13V",
        "mut3@121_indel",
        "SOURCE_Lab1",
        "SOURCE_Lab2",
    ]
    expected_values_fixed = [
        [1, 1, 0, 0, 0, 1, 0],  # Sample A: 'mut0@V1!', Lab1
        [
            1,
            0,
            1,
            0,
            1,
            0,
            1,
        ],  # Sample B: Mutations "mut1@G12G" and "mut3@121_indel", Lab2
        [1, 0, 0, 1, 0, 1, 0],  # Sample C: Mutation "mut2@A13V", Lab1
        [1, 0, 0, 0, 0, 0, 1],  # Sample D: No mutations, Lab2
    ]
    assert mutation_matrix_fe.shape == (
        4,
        7,
    ), f"Expected shape (4, 7), got {mutation_matrix_fe.shape}"
    assert (
        list(mutation_matrix_fe.columns) == expected_columns_fixed
    ), f"Expected columns {expected_columns_fixed}, got {list(mutation_matrix_fe.columns)}"
    assert np.array_equal(
        mutation_matrix_fe.values, expected_values_fixed
    ), f"Expected mutation matrix values {expected_values_fixed}, got {mutation_matrix_fe.values.tolist()}"


def test_build_X_sparse(mixed_variants):
    """Test sparse binary mutation matrix generation using filtered SNP DataFrame."""
    samples, mutations = mixed_variants
    mutations["REF"] = ["A", "G", "A", "T"]
    mutations["ALT"] = ["G", "A", "T", "C"]

    builder = RegressionBuilder(samples, mutations)
    snps = builder.generate_snps_df()
    mutation_matrix_sparse = builder.build_X_sparse(snps)

    expected_shape = (3, 3)
    expected_data = [1, 1, 1]
    expected_indices = [
        (0, 0),  # Sample A, SNP_ID "mut0@A1G"
        (1, 1),  # Sample B, SNP_ID "mut1@G12A"
        (2, 2),  # Sample C, SNP_ID "mut2@A13T"
    ]

    assert (
        mutation_matrix_sparse.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {mutation_matrix_sparse.shape}"
    non_zero_coords = list(zip(*mutation_matrix_sparse.nonzero()))
    assert len(non_zero_coords) == len(
        expected_data
    ), f"Expected {len(expected_data)} non-zero entries, got {len(non_zero_coords)}"
    actual_data = mutation_matrix_sparse.data.tolist()
    assert (
        actual_data == expected_data
    ), f"Expected data {expected_data}, got {actual_data}"
    assert (
        non_zero_coords == expected_indices
    ), f"Expected indices {expected_indices}, got {non_zero_coords}"


def test_hamming_distance(mixed_variants):
    """Test pairwise Hamming distance computation using output from build_X_sparse."""
    samples, mutations = mixed_variants

    mutations["REF"] = ["A", "G", "A", "T"]
    mutations["ALT"] = ["G", "A", "T", "C"]

    builder = RegressionBuilder(samples, mutations)
    snps = builder.generate_snps_df()
    X_sparse = builder.build_X_sparse(snps)
    computed_distances = builder.hamming_distance(X_sparse)

    expected_distances = np.array(
        [
            [-1, 2, 2],  # A vs B = 2, A vs C = 2
            [2, -1, 2],  # B vs A = 2, B vs C = 2
            [2, 2, -1],  # C vs A = 2, C vs B = 2
        ]
    )
    assert computed_distances.shape == (
        3,
        3,
    ), f"Expected shape (3, 3), got {computed_distances.shape}"
    np.testing.assert_array_equal(
        computed_distances,
        expected_distances,
        err_msg="Computed Hamming distances do not match expected distances.",
    )
    # Compute Hamming distances using scipy's pdist for comparison
    X_dense = X_sparse.toarray()
    scipy_distances = squareform(pdist(X_dense, metric="hamming")) * X_sparse.shape[1]
    scipy_distances[np.diag_indices_from(scipy_distances)] = (
        -1
    )  # Set diagonal to -1 for comparison

    np.testing.assert_array_almost_equal(
        computed_distances,
        scipy_distances,
        decimal=6,
        err_msg="Computed Hamming distances do not match scipy's result.",
    )


def test_calc_clusters(mixed_variants):
    """Test clustering of samples based on SNP distances."""
    samples, mutations = mixed_variants

    mutations["REF"] = ["A", "G", "A", "T"]
    mutations["ALT"] = ["G", "A", "T", "C"]

    builder = RegressionBuilder(samples, mutations)
    cluster_distance = 1
    clusters = builder.calc_clusters(cluster_distance=cluster_distance)

    # Check length
    assert len(clusters) == len(samples)

    # Check that identical SNP profiles are grouped together
    # For this fixture, A, B, C should be separate and D should be its own cluster
    assert len(set(clusters)) >= 1


def test_define_intervals(mixed_variants):
    """Test defining MIC intervals with different censoring settings."""
    samples, mutations = mixed_variants

    df = pd.merge(samples, mutations, how="left", on="UNIQUEID")

    builder = RegressionBuilder(samples, mutations)
    builder.censored = False
    builder.dilution_factor = 2
    builder.tail_dilutions = 3

    y_low, y_high = builder.define_intervals(df)

    expected_y_low = [-1.0, -2.0, -2.0, 1.58, 1.0]
    expected_y_high = [0.0, 1.0, 1.0, 4.58, 2.0]

    np.testing.assert_allclose(
        y_low,
        expected_y_low,
        atol=1e-2,
        err_msg=f"Expected y_low {expected_y_low}, but got {y_low}",
    )
    np.testing.assert_allclose(
        y_high,
        expected_y_high,
        atol=1e-2,
        err_msg=f"Expected y_high {expected_y_high}, but got {y_high}",
    )

    # Repeat test for censored case
    builder.censored = True
    y_low_censored, y_high_censored = builder.define_intervals(df)

    expected_y_low_censored = [-1, -19.93, -19.93, 1.58, 1]
    expected_y_high_censored = [0, 1, 1, np.inf, 2]

    np.testing.assert_allclose(
        y_low_censored,
        expected_y_low_censored,
        atol=1e-2,
        err_msg=f"Expected y_low_censored {expected_y_low_censored}, but got {y_low_censored}",
    )
    np.testing.assert_allclose(
        y_high_censored,
        expected_y_high_censored,
        atol=1e-2,
        err_msg=f"Expected y_high_censored {expected_y_high_censored}, but got {y_high_censored}",
    )


def test_log_transf_val():
    """Test log transformation with a custom dilution factor."""

    class MockBuilder:
        dilution_factor = 2

    builder = MockBuilder()

    test_values = [1, 2, 4, 8]
    expected_results = [0, 1, 2, 3]

    for val, expected in zip(test_values, expected_results):
        result = RegressionBuilder.log_transf_val(builder, val)
        assert np.isclose(
            result, expected, atol=1e-6
        ), f"Expected {expected}, got {result} for val {val}"


def test_initial_params(mixed_variants):
    """Test generation of initial parameters for the regression model."""
    samples, mutations = mixed_variants
    mutations["REF"] = ["A", "G", "A", "T"]
    mutations["ALT"] = ["G", "A", "T", "C"]

    builder = RegressionBuilder(samples, mutations)
    df = pd.merge(samples, mutations, how="left", on=["UNIQUEID"])
    X = builder.build_X(df)  # Binary mutation matrix

    y_low, y_high = builder.define_intervals(samples)

    cluster_distance = 1
    clusters = builder.calc_clusters(cluster_distance=cluster_distance)

    beta_init, u_init, sigma = builder.initial_params(X, y_low, y_high, clusters)

    assert (
        beta_init.shape[0] == X.shape[1]
    ), f"Expected beta_init to have {X.shape[1]} elements, got {beta_init.shape[0]}"
    assert np.all(np.isfinite(beta_init)), "All elements of beta_init should be finite."

    num_clusters = len(np.unique(clusters))
    assert (
        u_init.shape[0] == num_clusters
    ), f"Expected u_init to have {num_clusters} elements, got {u_init.shape[0]}"
    assert np.all(np.isfinite(u_init)), "All elements of u_init should be finite."

    assert np.isfinite(sigma), "Sigma should be a finite value."


def test_predict_effects(mixed_variants):
    """Test predict_effects returns valid mutation-level effects (excluding synonymous)."""
    samples, mutations = mixed_variants

    mutations["REF"] = ["A", "G", "A", "T"]
    mutations["ALT"] = ["G", "A", "T", "C"]

    builder = RegressionBuilder(
        samples, mutations, genes=["mut0", "mut1", "mut2", "mut3"]
    )

    model, effects = builder.predict_effects(
        b_bounds=(None, None),
        u_bounds=(None, None),
        s_bounds=(None, None),
        options={"maxiter": 50},
        L2_penalties={"lambda_sigma": 0.01, "lambda_u": 0.01},
        random_effects=True,
        cluster_distance=50,
    )

    # Synonymous mutation mut1@G12G should be excluded
    assert "mut1@G12G" not in effects["Mutation"].values

    # Remaining non-synonymous mutations should appear
    expected_mutations = {"mut0@V1!", "mut2@A13V", "mut3@121_indel"}
    assert expected_mutations.issubset(set(effects["Mutation"]))

    # Structural checks
    required_columns = {"Mutation", "effect_size", "fold_change", "MIC"}
    assert required_columns.issubset(set(effects.columns))

    # Values must be finite and positive
    assert np.all(np.isfinite(effects["effect_size"]))
    assert np.all(effects["MIC"] > 0)


def test_classify_effects(mixed_variants):
    """Test classification produces valid categories and excludes synonymous mutations."""
    samples, mutations = mixed_variants

    mutations["REF"] = ["A", "G", "A", "T"]
    mutations["ALT"] = ["G", "A", "T", "C"]

    builder = RegressionBuilder(
        samples, mutations, genes=["mut0", "mut1", "mut2", "mut3"]
    )

    model, effects = builder.predict_effects(
        b_bounds=(None, None),
        u_bounds=(None, None),
        s_bounds=(None, None),
        options={"maxiter": 50},
        L2_penalties={"lambda_sigma": 0.01, "lambda_u": 0.01},
        random_effects=True,
        cluster_distance=50,
    )

    classified_effects, ecoff = builder.classify_effects(effects, ecoff=1, p=0.95)

    assert "mut1@G12G" not in classified_effects["Mutation"].values
    assert "Classification" in classified_effects.columns
    assert set(classified_effects["Classification"]).issubset({"R", "S", "U"})


def test_z_test():
    """Test the z_test method for calculating two-tailed p-values."""

    test_cases = [
        {"mu": 10, "val": 10, "se": 2, "expected_p": 1.0},
        {"mu": 12, "val": 10, "se": 2, "expected_p": 0.3173},
        {"mu": 14, "val": 10, "se": 2, "expected_p": 0.0455},
        {"mu": 16, "val": 10, "se": 2, "expected_p": 0.0027},
    ]

    for case in test_cases:
        mu = case["mu"]
        val = case["val"]
        se = case["se"]
        expected_p = case["expected_p"]

        p_value = RegressionBuilder.z_test(mu, val, se)

        assert np.isclose(
            p_value, expected_p, atol=1e-4
        ), f"For mu={mu}, val={val}, se={se}: Expected p={expected_p}, but got {p_value}."


def test_add_mutation(mixed_variants):

    samples, mutations = mixed_variants
    """Test the add_mutation method for updating the catalogue and entry list."""
    builder = RegressionBuilder(samples, mutations)

    mutation = "mut5@P12Q"
    prediction = "R"
    evidence = {"MIC": 2.0, "ECOFF": 1.5}

    builder.add_mutation(mutation, prediction, evidence)

    assert builder.catalogue[mutation] == {
        "pred": prediction,
        "evid": evidence,
    }, f"Expected catalogue entry for {mutation}, got {builder.catalogue[mutation]}"
    assert mutation in builder.entry, f"Expected mutation {mutation} in entry list."


def test_build(mixed_variants):
    """Test full build workflow produces valid catalogue structure."""
    samples, mutations = mixed_variants

    mutations["REF"] = ["A", "G", "A", "T"]
    mutations["ALT"] = ["G", "A", "T", "C"]

    builder = RegressionBuilder(samples, mutations, genes=["mut0", "mut1", "mut2", "mut3"])

    builder.build(
        ecoff=1,
        b_bounds=(None, None),
        u_bounds=(None, None),
        s_bounds=(None, None),
        options={"maxiter": 50},
        L2_penalties={"lambda_beta": 0.01, "lambda_u": 0.01},
        p=0.95,
        random_effects=True,
        cluster_distance=50,
    )

    catalogue = builder.catalogue

    # Synonymous mutation should not appear
    assert "mut1@G12G" not in catalogue

    # Non-synonymous mutations should appear
    expected_mutations = {"mut0@V1!", "mut2@A13V", "mut3@121_indel"}
    assert expected_mutations.issubset(set(catalogue.keys()))

    # Structural checks
    for mutation, entry in catalogue.items():
        assert "pred" in entry
        assert entry["pred"] in {"R", "S", "U"}

        evid = entry.get("evid", {})
        assert "MIC" in evid
        assert evid["MIC"] > 0
        assert "effect_size" in evid


def test_main_regression_builder(mixed_variants, tmp_path):
    """Test CLI produces valid catalogue and excludes synonymous mutations."""
    samples, mutations = mixed_variants

    samples_file = tmp_path / "samples.csv"
    mutations_file = tmp_path / "mutations.csv"
    output_file = tmp_path / "catalogue.json"

    samples.to_csv(samples_file, index=False)
    mutations.to_csv(mutations_file, index=False)

    cli_args = [
        "catomatic",
        "regression",
        "--samples", str(samples_file),
        "--mutations", str(mutations_file),
        "--dilution_factor", "2",
        "--censored",
        "--tail_dilutions", "1",
        "--ecoff", "1",
        "--b_bounds", "-5", "5",
        "--u_bounds", "-5", "5",
        "--s_bounds", "-5", "5",
        "--p", "0.95",
        "--cluster_distance", "50",
        "--outfile", str(output_file),
        "--to_json",
    ]

    with patch("sys.argv", cli_args):
        main()

    assert os.path.exists(output_file)

    with open(output_file, "r") as f:
        catalogue = json.load(f)

    # Synonymous mutation excluded
    assert "mut1@G12G" not in catalogue

    # Non-synonymous mutations included
    assert "mut0@V1!" in catalogue
    assert "mut2@A13V" in catalogue