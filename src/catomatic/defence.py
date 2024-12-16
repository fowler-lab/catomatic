import os
import pandas as pd
import warnings


def soft_assert(condition, message="Warning!"):
    """
    Issues a warning if the condition is not met.
    """
    if not condition:
        warnings.warn(message, stacklevel=2)


def validate_binary_build_inputs(
    samples,
    mutations,
    seed,
    FRS,
    test,
    background,
    p,
    tails,
    record_ids,
):
    """
    Validates the input parameters and raises errors or warnings as necessary.
    """
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
