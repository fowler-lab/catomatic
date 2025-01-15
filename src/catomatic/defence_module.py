import os
import pandas as pd
import warnings


def soft_assert(condition, message="Warning!"):
    """
    Issues a warning if the condition is not met.
    """
    if not condition:
        warnings.warn(message, stacklevel=2)


def validate_binary_init(
    samples,
    mutations,
    seed,
    FRS,
):
    # Check samples and mutations dataframes
    assert all(
        column in samples.columns for column in ["UNIQUEID", "PHENOTYPE"]
    ), "Input df must contain columns UNIQUEID and PHENOTYPE"

    assert all(
        column in mutations.columns for column in ["UNIQUEID", "MUTATION"]
    ), "Input df must contain columns UNIQUEID and MUTATION"

    assert samples.UNIQUEID.nunique() == len(
        samples.UNIQUEID
    ), "Each sample should have only 1 phenotype"

    assert all(
        i in ["R", "S"] for i in samples.PHENOTYPE
    ), "Binary phenotype values must either be R or S"

    assert (
        len(pd.merge(samples, mutations, on=["UNIQUEID"], how="left")) > 0
    ), "No UNIQUEIDs for mutations match UNIQUEIDs for samples!"

    if seed is not None:
        assert isinstance(
            seed, list
        ), "The 'seed' parameter must be a list of neutral (susceptible) mutations."
        soft_assert(
            all(s in mutations.MUTATION.values for s in seed),
            "Not all seeds are represented in mutations table, are you sure the grammar is correct?",
        )

    if FRS is not None:
        assert isinstance(FRS, float), "FRS must be a float"
        assert (
            "FRS" in mutations.columns
        ), 'The mutations df must contain an "FRS" column to filter by FRS'


def validate_binary_build_inputs(
    test,
    background,
    p,
    tails,
    record_ids,
):
    """
    Validates the input parameters and raises errors or warnings as necessary.
    """

    assert isinstance(record_ids, bool), "record_ids parameter must be of type bool."

    if test is not None:
        assert test in [
            None,
            "Binomial",
            "Fisher",
        ], "The test must be None, Binomial or Fisher"
        if test == "Binomial":
            assert background is not None and isinstance(
                background, float
            ), "If using a binomial test, an assumed background resistance rate (0-1) must be specified"
            assert p < 1, "The p value for statistical testing must be 0 < p < 1"
        elif test == "Fisher":
            assert p < 1, "The p value for statistical testing must be 0 < p < 1"

        assert isinstance(tails, str) and tails in [
            "two",
            "one",
        ], "tails must either be 'one' or 'two'"


def validate_regression_init(
    samples,
    mutations,
    genes,
    dilution_factor,
    censored,
    tail_dilutions,
    FRS,
    seed,
):
    # Check samples and mutations dataframes
    assert all(
        column in samples.columns for column in ["UNIQUEID", "MIC"]
    ), "Input df must contain columns UNIQUEID and MIC"

    assert all(
        column in mutations.columns for column in ["UNIQUEID", "MUTATION"]
    ), "Input df must contain columns UNIQUEID and MUTATION"

    assert samples.UNIQUEID.nunique() == len(
        samples.UNIQUEID
    ), "Each sample should have only 1 MIC reading"

    if len(genes) > 0:
        # Ensure element-wise splitting of 'MUTATION' column
        assert any(
            mutations["MUTATION"].str.split("@").str[0].isin(genes)
        ), "No mutations match the specified genes."

    assert samples["MIC"].notna().all(), "MIC column contains NaN values."

    assert isinstance(
        dilution_factor, (int, float)
    ), "Dilution factor must be an integer or float."
    assert dilution_factor > 0, "Dilution factor must be greater than zero."

    assert isinstance(
        censored, bool
    ), "Censored must be a boolean value (True or False)."

    assert isinstance(tail_dilutions, int), "Tail dilutions must be an integer."
    assert tail_dilutions >= 0, "Tail dilutions must be zero or a positive integer."

    if FRS is not None:
        assert isinstance(FRS, (int, float)), "FRS must be a float or integer."
        assert (
            "FRS" in mutations.columns
        ), 'The mutations DataFrame must contain an "FRS" column to use FRS filtering.'

    assert not samples.empty, "Samples DataFrame must not be empty."

    assert set(mutations["UNIQUEID"]).issubset(
        set(samples["UNIQUEID"])
    ), "All UNIQUEID values in mutations must exist in samples."

    assert isinstance(seed, int), "The random seed must be an integer"


def validate_regression_predict_inputs(
    columns,
    b_bounds,
    u_bounds,
    s_bounds,
    options,
    L2_penalties,
    fixed_effects,
    random_effects,
    cluster_distance,
    genes,
):
    for bounds, name in zip(
        [b_bounds, u_bounds, s_bounds], ["b_bounds", "u_bounds", "s_bounds"]
    ):
        if bounds is not None:
            assert (
                isinstance(bounds, (tuple, list)) and len(bounds) == 2
            ), f"{name} must be a tuple with two elements (min, max)."
            assert all(
                x is None or isinstance(x, (int, float)) for x in bounds
            ), f"{name} must contain only numeric values or None."
            if all(x is not None for x in bounds):
                assert (
                    bounds[0] <= bounds[1]
                ), f"Invalid range in {name}: min cannot be greater than max."

    if options is not None:
        assert isinstance(
            options, dict
        ), "Options must be a dictionary of scipy minimise arguments."

    if L2_penalties is not None:
        assert isinstance(L2_penalties, dict), "L2_penalties must be a dictionary."
        valid_keys = {"lambda_beta", "lambda_u", "lambda_sigma"}
        assert set(L2_penalties.keys()).issubset(
            valid_keys
        ), f"L2_penalties keys must be a subset of {valid_keys}."
        for key, value in L2_penalties.items():
            assert isinstance(
                value, (int, float)
            ), f"{key} in L2_penalties must be numeric."
            assert value >= 0, f"{key} in L2_penalties must be non-negative."

    assert isinstance(
        random_effects, bool
    ), "Random effects must be a boolean value (True or False)."

    if random_effects:
        assert len(genes) > 0, (
            "If calculating random effect SNP distance clusters, "
            "must instantiate with a whole genome mutations table (for clustering), "
            "and a list of RAV genes to filter this by (for regression)"
        )
        assert (
            isinstance(cluster_distance, int) and cluster_distance > 0
        ), "Cluster distance must be a number greater than 0."

    if fixed_effects is not None:
        assert isinstance(
            fixed_effects, list
        ), "Fixed effects must be a list of column names"
        assert all(fe in columns for fe in fixed_effects), "One or more fixed effects do not exist in input data"



def validate_regression_classify_inputs(
    ecoff,
    percentile,
    p,
):

    if ecoff is not None:
        assert isinstance(ecoff, (int, float)), "ECOFF must be a numeric value."
        assert ecoff > 0, "ECOFF must be a positive value."

    assert isinstance(percentile, (int, float)), "Percentile must be numeric."
    assert 0 < percentile <= 100, "Percentile must be between 1 and 100."

    assert isinstance(p, (int, float)), "Significance level (p) must be numeric."
    assert 0 < p < 1, "Significance level (p) must be between 0 and 1."


def validate_build_piezo_inputs(
    genbank_ref,
    catalogue_name,
    version,
    drug,
    wildcards,
    grammar,
    values,
    public,
    for_piezo,
    json_dumps,
    include_U,
):
    """
    Validates inputs for the build_piezo method to ensure they meet the expected types and values.
    """
    # Check string inputs
    assert isinstance(genbank_ref, str), "genbank_ref must be a string."
    assert isinstance(catalogue_name, str), "catalogue_name must be a string."
    assert isinstance(version, str), "version must be a string."
    assert isinstance(drug, str), "drug must be a string."

    # Check wildcards: should be dict or a valid file path
    assert isinstance(
        wildcards, (dict, str)
    ), "wildcards must be a dict or a file path (str)."
    if isinstance(wildcards, str):
        assert os.path.exists(
            wildcards
        ), "If wildcards is a file path, the file must exist."

    # Check grammar
    assert grammar in ["GARC1"], "Only 'GARC1' grammar is currently supported."

    # Check values
    assert values == "RUS", "Only 'RUS' values are currently supported."

    # Check boolean inputs
    assert isinstance(public, bool), "public must be a boolean."
    assert isinstance(for_piezo, bool), "for_piezo must be a boolean."
    assert isinstance(json_dumps, bool), "json_dumps must be a boolean."
    assert isinstance(include_U, bool), "include_U must be a boolean."


def validate_ecoff_inputs(
    samples, mutations, dilution_factor, censored, tail_dilutions
):
    """Validates inputs for the ECOFF generator initialization."""

    assert isinstance(samples, pd.DataFrame), "samples must be a pandas DataFrame."
    assert isinstance(mutations, pd.DataFrame), "mutations must be a pandas DataFrame."

    # Check required columns in samples
    assert all(
        column in samples.columns for column in ["UNIQUEID", "MIC"]
    ), "Input samples must contain columns 'UNIQUEID' and 'MIC'"

    # Check required columns in mutations
    assert all(
        column in mutations.columns for column in ["UNIQUEID", "MUTATION"]
    ), "Input mutations must contain columns 'UNIQUEID' and 'MUTATION'"

    # Validate dilution_factor
    assert (
        isinstance(dilution_factor, int) and dilution_factor > 0
    ), "dilution_factor must be a positive integer."

    # Validate censored flag
    assert isinstance(
        censored, bool
    ), "censored must be a boolean value (True or False)."

    # Validate tail_dilutions if censored is False
    if not censored:
        assert (
            isinstance(tail_dilutions, int) and tail_dilutions > 0
        ), "When censored is False, tail_dilutions must be a positive integer or specified."
