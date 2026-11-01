from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Literal, List

import pandas as pd


TestMode = Optional[Literal["Binomial", "Fisher"]]
Tails = Literal["one", "two"]


def soft_assert(
    condition: bool,
    message: str = "Warning!",
    *,
    category: type[Warning] = UserWarning,
) -> None:
    """
    Emit a warning if a condition is not met.

    Args:
        condition: Condition to evaluate.
        message: Warning message if condition is False.
        category: Warning class to emit (defaults to UserWarning).

    Returns:
        None
    """
    if not condition:
        warnings.warn(message, category=category, stacklevel=2)


def _require_columns(df: pd.DataFrame, required: Sequence[str], *, name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} must contain columns {list(required)}; missing {missing}."
        )


def _require_unique(df: pd.DataFrame, column: str, *, name: str) -> None:
    if df[column].nunique(dropna=False) != len(df[column]):
        raise ValueError(f"{name} must have unique values in column '{column}'.")


def validate_binary_init(
    samples: pd.DataFrame,
    mutations: pd.DataFrame,
    seed: Optional[list[str]],
    frs: Optional[float],
) -> None:
    """
    Validate inputs for BinaryBuilder.__init__.

    Args:
        samples: DataFrame with ['UNIQUEID', 'PHENOTYPE'].
        mutations: DataFrame with ['UNIQUEID', 'MUTATION'] and optional 'FRS'.
        seed: Optional list of seeded mutations.
        frs: Optional FRS threshold.

    Returns:
        None
    """
    _require_columns(samples, ["UNIQUEID", "PHENOTYPE"], name="samples")
    _require_columns(mutations, ["UNIQUEID", "MUTATION"], name="mutations")

    _require_unique(samples, "UNIQUEID", name="samples")

    if not set(samples["PHENOTYPE"]).issubset({"R", "S"}):
        raise ValueError("Binary phenotype values must be either 'R' or 'S'.")

    if pd.merge(
        samples[["UNIQUEID"]], mutations[["UNIQUEID"]], on="UNIQUEID", how="inner"
    ).empty:
        raise ValueError("No UNIQUEIDs for mutations match UNIQUEIDs for samples.")

    if seed is not None:
        if not isinstance(seed, list) or not all(isinstance(s, str) for s in seed):
            raise TypeError(
                "seed must be a list[str] of neutral (susceptible) mutations."
            )
        soft_assert(
            all(s in set(mutations["MUTATION"]) for s in seed),
            "Not all seeds are represented in mutations table; confirm grammar and mutation identifiers.",
        )

    if frs is not None:
        if not isinstance(frs, float):
            raise TypeError("frs must be a float.")
        _require_columns(mutations, ["FRS"], name="mutations")


def validate_binary_build_inputs(
    test: TestMode,
    background: Optional[float],
    p: float,
    tails: Tails,
    record_ids: bool,
) -> None:
    """
    Validate inputs for BinaryBuilder.build.

    Args:
        test: 'Binomial', 'Fisher', or None.
        background: Background resistance rate for binomial test.
        p: Confidence parameter (0 < p < 1); builder typically uses 1 - p internally.
        tails: 'one' or 'two'.
        record_ids: Whether to store UNIQUEIDs in evidence records.

    Returns:
        None
    """
    if not isinstance(record_ids, bool):
        raise TypeError("record_ids must be a bool.")

    if test not in (None, "Binomial", "Fisher"):
        raise ValueError("test must be None, 'Binomial', or 'Fisher'.")

    if not isinstance(p, (int, float)) or not (0 < p < 1):
        raise ValueError("p must satisfy 0 < p < 1.")

    if tails not in ("one", "two"):
        raise ValueError("tails must be either 'one' or 'two'.")

    if test == "Binomial":
        if background is None or not isinstance(background, (int, float)):
            raise TypeError(
                "background must be supplied as a float if test == 'Binomial'."
            )
        if not (0 <= float(background) <= 1):
            raise ValueError("background must be in [0, 1].")


def validate_regression_init(
    samples: pd.DataFrame,
    mutations: pd.DataFrame,
    genes: List[str],
    dilution_factor: float,
    censored: bool,
    tail_dilutions: int,
    frs: Optional[float],
    seed: int,
) -> None:
    """
    Validate inputs for RegressionBuilder.__init__.

    Args:
        samples: DataFrame with ['UNIQUEID', 'MIC'].
        mutations: DataFrame with ['UNIQUEID', 'MUTATION'] and optional 'FRS'.
        genes: Target gene list; if non-empty, mutations must overlap.
        dilution_factor: Positive scaling base.
        censored: Whether MIC data are censored at extremes.
        tail_dilutions: Tail extension in dilutions when not censored.
        frs: Optional threshold.
        seed: Random seed for initialisation.

    Returns:
        None
    """
    _require_columns(samples, ["UNIQUEID", "MIC"], name="samples")
    _require_columns(mutations, ["UNIQUEID", "MUTATION"], name="mutations")

    _require_unique(samples, "UNIQUEID", name="samples")

    if samples["MIC"].isna().any():
        raise ValueError("MIC column contains NaN values.")

    if not isinstance(dilution_factor, (int, float)) or dilution_factor <= 0:
        raise ValueError("dilution_factor must be a positive number.")

    if not isinstance(censored, bool):
        raise TypeError("censored must be a bool.")

    if not isinstance(tail_dilutions, int) or tail_dilutions < 0:
        raise ValueError("tail_dilutions must be a non-negative integer.")

    if frs is not None:
        if not isinstance(frs, (int, float)):
            raise TypeError("frs must be numeric.")
        _require_columns(mutations, ["FRS"], name="mutations")

    if samples.empty:
        raise ValueError("samples must not be empty.")

    if not set(mutations["UNIQUEID"]).issubset(set(samples["UNIQUEID"])):
        raise ValueError("All UNIQUEID values in mutations must exist in samples.")

    if not isinstance(seed, int):
        raise TypeError("seed must be an int.")

    if len(genes) > 0:
        if not all(isinstance(g, str) for g in genes):
            raise TypeError("genes must be a sequence of strings.")
        # Ensure MUTATION is string-like for splitting
        if not pd.api.types.is_string_dtype(mutations["MUTATION"]):
            raise TypeError(
                "mutations['MUTATION'] must be string-like when genes are provided."
            )
        gene_part = mutations["MUTATION"].astype(str).str.split("@").str[0]
        if not gene_part.isin(list(genes)).any():
            raise ValueError("No mutations match the specified genes.")


from typing import Any, Mapping, Optional, Sequence, Tuple, List

def validate_regression_predict_inputs(
    columns: Sequence[str],
    b_bounds: Tuple[Optional[float], Optional[float]],
    u_bounds: Tuple[Optional[float], Optional[float]],
    s_bounds: Tuple[Optional[float], Optional[float]],
    options: Optional[Mapping[str, Any]],
    L2_penalties: Optional[Mapping[str, Any]],
    fixed_effects: Optional[Sequence[str]],
    random_effects: bool,
    cluster_distance: int,
    genes: Sequence[str],
) -> None:
    """
    Validate inputs for RegressionBuilder.predict_effects.

    Args:
        columns: samples df columns.
        b_bounds/u_bounds/s_bounds: (min, max) bounds, each element numeric or None.
        options: Optimizer options mapping.
        L2_penalties: Regularization mapping.
        fixed_effects: Optional list of fixed-effect columns that must be in `columns`.
        random_effects: Whether clustering is enabled.
        cluster_distance: Positive int distance threshold.
        genes: Required if random_effects is True.

    Returns:
        None
    """

    for bounds, name in (
        (b_bounds, "b_bounds"),
        (u_bounds, "u_bounds"),
        (s_bounds, "s_bounds"),
    ):
        # Ensure shape/type
        if not (isinstance(bounds, (tuple, list)) and len(bounds) == 2):
            raise TypeError(f"{name} must be a (min, max) tuple.")
        # Ensure elements are numeric or None
        if not all(x is None or isinstance(x, (int, float)) for x in bounds):
            raise TypeError(f"{name} must contain only numeric values or None.")

        lo, hi = bounds 
        if (lo is not None) and (hi is not None):
            if lo > hi:
                raise ValueError(f"Invalid range in {name}: min cannot be greater than max.")

    if options is not None and not isinstance(options, Mapping):
        raise TypeError("options must be a mapping of optimizer arguments.")

    if L2_penalties is not None:
        if not isinstance(L2_penalties, Mapping):
            raise TypeError("L2_penalties must be a mapping.")
        valid_keys = {"lambda_beta", "lambda_u", "lambda_sigma"}
        if not set(L2_penalties.keys()).issubset(valid_keys):
            raise ValueError(f"L2_penalties keys must be a subset of {valid_keys}.")
        for key, value in L2_penalties.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"{key} in L2_penalties must be numeric.")
            if value < 0:
                raise ValueError(f"{key} in L2_penalties must be non-negative.")

    if not isinstance(random_effects, bool):
        raise TypeError("random_effects must be a bool.")

    if random_effects:
        if len(genes) == 0:
            raise ValueError(
                "If random_effects is True, genes must be provided (RAV genes for regression; "
                "whole-genome mutations required for clustering)."
            )
        if not isinstance(cluster_distance, int) or cluster_distance <= 0:
            raise ValueError("cluster_distance must be a positive integer.")

    if fixed_effects is not None:
        if not isinstance(fixed_effects, (list, tuple)):
            raise TypeError("fixed_effects must be a sequence of column names.")
        missing = [fe for fe in fixed_effects if fe not in columns]
        if missing:
            raise ValueError(
                f"One or more fixed effects do not exist in input data: {missing}."
            )


def validate_regression_classify_inputs(
    ecoff: float,
    p: float,
) -> None:
    """
    Validate inputs for regression effect classification.

    Args:
        ecoff:  ECOFF (MIC scale).
        p: Confidence parameter (0 < p < 1).

    Returns:
        None
    """

    if not isinstance(ecoff, (int, float)):
        raise TypeError("ecoff must be numeric.")
    if ecoff <= 0:
        raise ValueError("ecoff must be positive.")

    if not isinstance(p, (int, float)) or not (0 < p < 1):
        raise ValueError("p must satisfy 0 < p < 1.")


def validate_build_piezo_inputs(
    genbank_ref: str,
    catalogue_name: str,
    version: str,
    drug: str,
    wildcards: Mapping[str, Any] | str | Path,
    grammar: str,
    values: str,
    public: bool,
    for_piezo: bool,
    json_dumps: bool,
    include_U: bool,
) -> None:
    """
    Validate inputs for PiezoExporter.build_piezo.

    Args:
        genbank_ref: GenBank reference identifier.
        catalogue_name: Catalogue name.
        version: Catalogue version.
        drug: Drug.
        wildcards: Mapping or a path to JSON.
        grammar: Must be 'GARC1'.
        values: Must be 'RUS'.
        public: Public/export mode.
        for_piezo: Whether to add placeholders.
        json_dumps: Whether to JSON encode evidence columns.
        include_U: Whether to include non-placeholder 'U' entries.

    Returns:
        None
    """
    for s, name in (
        (genbank_ref, "genbank_ref"),
        (catalogue_name, "catalogue_name"),
        (version, "version"),
        (drug, "drug"),
    ):
        if not isinstance(s, str) or not s:
            raise TypeError(f"{name} must be a non-empty string.")

    if isinstance(wildcards, (str, Path)):
        path = Path(wildcards)
        if not path.exists():
            raise FileNotFoundError("If wildcards is a file path, the file must exist.")
    elif not isinstance(wildcards, Mapping):
        raise TypeError("wildcards must be a mapping or a file path.")

    if grammar != "GARC1":
        raise ValueError("Only 'GARC1' grammar is currently supported.")

    if values != "RUS":
        raise ValueError("Only 'RUS' values are currently supported.")

    for b, name in (
        (public, "public"),
        (for_piezo, "for_piezo"),
        (json_dumps, "json_dumps"),
        (include_U, "include_U"),
    ):
        if not isinstance(b, bool):
            raise TypeError(f"{name} must be a bool.")
