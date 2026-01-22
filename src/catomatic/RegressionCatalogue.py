import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.stats import norm
from .PiezoTools import PiezoExporter
from .defence_module import (
    validate_regression_init,
    validate_regression_predict_inputs,
    validate_regression_classify_inputs,
)
from typing import Any, Optional, Sequence, Tuple
from intreg.meintreg import MeIntReg
from sklearn.cluster import AgglomerativeClustering


class RegressionBuilder(PiezoExporter):
    """
    Builds a mutation catalogue compatible with Piezo in a standardized format.

    Regression labels underpin a distributional modelling approach.

    MICs are treated as intervals to fit a regression curve assuming a Gaussian distribution.
    Instantiation constructs the builder object (sample/mutation tables + configuration), and
    `build()` orchestrates fitting, effect extraction, and classification into catalogue entries.

    Parameters:
        samples (pd.DataFrame | str): A DataFrame (or path to CSV) containing sample identifiers and MICs.
                                      Required columns: ['UNIQUEID', 'MIC'].

        mutations (pd.DataFrame | str): A DataFrame (or path to CSV) containing mutations for each sample.
                                        Required columns: ['UNIQUEID', 'MUTATION'].
                                        Optional columns: ['frs', 'REF', 'ALT', 'SNP_ID'].

        genes (list[str], optional): A list of target genes. If supplied, only mutations whose gene component
                                     (the substring before '@') is in this list are modelled. If non-target
                                     genes are present in the mutations table and population-structure clustering
                                     is enabled, this list should be supplied to avoid unintended clustering inputs.

        dilution_factor (int, optional): Base for MIC dilution scaling (default 2; doubling series).

        censored (bool, optional): Whether MIC interval tails are treated as censored (default True).
                                   If False, intervals are extended by `tail_dilutions`.

        tail_dilutions (int, optional): Number of additional dilutions to extend interval tails when
                                        `censored` is False.

        frs (float, optional): Fraction read support threshold used to filter mutations (default None).
                               Note this also affects SNP clustering inputs.

        seed (int, optional): Random seed controlling only the initial parameter generator (default 0).
    """

    samples: pd.DataFrame
    mutations: pd.DataFrame
    catalogue: dict[str, dict[str, Any]]
    entry: list[str]

    genes: list[str]
    dilution_factor: int
    censored: bool
    tail_dilutions: int

    # set during prediction/build
    target_mutations: pd.DataFrame
    df: pd.DataFrame

    def __init__(
        self,
        samples: pd.DataFrame | str,
        mutations: pd.DataFrame | str,
        genes: Optional[list[str]] = None,
        dilution_factor: int = 2,
        censored: bool = True,
        tail_dilutions: int = 1,
        frs: Optional[float] = None,
        seed: int = 0,
    ) -> None:
        """
        Initialize the RegressionBuilder with sample and mutation tables.

        Args:
            samples: DataFrame or path to CSV with columns ['UNIQUEID', 'MIC'].
            mutations: DataFrame or path to CSV with columns ['UNIQUEID', 'MUTATION'] and optional metadata columns.
            genes: Optional list of target genes (see class docstring).
            dilution_factor: Dilution base used for MIC scaling.
            censored: Whether censoring is assumed for interval tails.
            tail_dilutions: Tail extension in dilutions if not censored.
            frs: Optional fraction read support threshold to filter mutation rows.
            seed: Random seed (only impacts the initial parameter generator).

        Returns:
            None
        """

        samples = pd.read_csv(samples) if isinstance(samples, str) else samples
        mutations = pd.read_csv(mutations) if isinstance(mutations, str) else mutations

        validate_regression_init(
            samples,
            mutations,
            genes or [],
            dilution_factor,
            censored,
            tail_dilutions,
            frs,
            seed,
        )

        if frs is not None:
            # note this will filter out mutations for clustering as well
            mutations = mutations[mutations.FRS >= frs]

        self.samples, self.mutations = samples, mutations

        self.genes = genes if genes is not None else []
        self.dilution_factor = dilution_factor
        self.censored = censored
        self.tail_dilutions = tail_dilutions
        np.random.seed(seed)

        # instantiate catalogue object
        self.catalogue = {}
        self.entry = []

    def build_X(
        self,
        df: pd.DataFrame,
        fixed_effects: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Build a binary mutation matrix X and optionally include fixed effects.

        Mutations are one-hot encoded as columns. If `fixed_effects` are supplied, they appended to X.

        Args:
            df: DataFrame containing at least ['UNIQUEID', 'MUTATION'] and optionally fixed-effect columns.
            fixed_effects: Optional list of column names in `df` to include as fixed effects.

        Returns:
            Binary mutation matrix indexed by UNIQUEID, with optional fixed effects appended.
        """
        ids = df.UNIQUEID.unique()

        # Create the binary mutation matrix
        X = pd.pivot_table(
            df[["UNIQUEID", "MUTATION"]],
            index="UNIQUEID",
            columns="MUTATION",
            aggfunc=lambda x: 1,  # Map presence to 1
            fill_value=0,  # Absence is 0
        ).reindex(ids, fill_value=0)

        if fixed_effects is not None:
            # Select the fixed effects columns and encode them properly
            fixed_effects_data = (
                df[["UNIQUEID"] + fixed_effects].drop_duplicates().set_index("UNIQUEID")
            )

            # One-hot encode the fixed effects
            fixed_effects_encoded = (
                pd.get_dummies(
                    fixed_effects_data,
                    columns=fixed_effects,
                    prefix=fixed_effects,  # Prefix helps to distinguish columns
                    drop_first=False,
                )
                .reindex(ids, fill_value=0)
                .astype(int)
            )

            # Combine the mutation matrix with the fixed effects
            X = pd.concat([X, fixed_effects_encoded], axis=1)

        return X

    @staticmethod
    def build_X_sparse(df: pd.DataFrame) -> csr_matrix:
        """
        Build a sparse binary mutation matrix for SNP IDs.

        Args:
            df: DataFrame containing ['UNIQUEID', 'SNP_ID'].

        Returns:
            Sparse binary matrix where rows are samples and columns are SNP IDs.
        """

        ids = df["UNIQUEID"].astype("category")
        mutations = df["SNP_ID"].astype("category")

        # Create a sparse matrix with 1 for presence
        row = ids.cat.codes
        col = mutations.cat.codes
        data = [1] * len(df)

        X = csr_matrix(
            (data, (row, col)),
            shape=(len(ids.cat.categories), len(mutations.cat.categories)),
        )

        return X

    @staticmethod
    def hamming_distance(
        X_sparse: csr_matrix,
        n_jobs: int = -1,
        block_size: int = 1000,
    ) -> np.ndarray:
        """
        Compute pairwise absolute Hamming distance for a sparse binary matrix.

        Args:
            X_sparse: Sparse binary matrix.
            n_jobs: Number of parallel jobs (-1 uses all available cores).
            block_size: Block size for chunked computation.

        Returns:
            Pairwise absolute Hamming distance matrix.
        """
        n_samples = X_sparse.shape[0]
        distances = np.zeros((n_samples, n_samples))

        def process_block(i, j):
            block_i = X_sparse[i : min(i + block_size, n_samples)]
            block_j = X_sparse[j : min(j + block_size, n_samples)]

            # compute intersection (dot product)
            intersect = block_i.dot(block_j.T)
            row_sums_i = block_i.sum(axis=1)
            row_sums_j = block_j.sum(axis=1).T
            union = row_sums_i + row_sums_j - intersect

            # calculate absolute hamming distance
            dist_block = union - 2 * intersect
            return i, j, dist_block

        # process blocks in parallel
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(process_block)(i, j)
            for i in range(0, n_samples, block_size)
            for j in range(i, n_samples, block_size)
        )

        # populate distance matrix
        for i, j, block_dist in results:
            rows = slice(i, min(i + block_size, n_samples))
            cols = slice(j, min(j + block_size, n_samples))
            distances[rows, cols] = block_dist
            if i != j:
                distances[cols, rows] = block_dist.T

        return distances

    def generate_snps_df(self) -> pd.DataFrame:
        """
        Generate a SNP-only DataFrame suitable for clustering, ensuring a 'SNP_ID' column exists.

        SNP rows are derived from self.mutations by excluding indels/ins/del/LOF/Z markers. If
        'SNP_ID' is not present, it is constructed from mutation/gene position plus REF/ALT.

        Returns:
            Filtered SNP DataFrame containing a 'SNP_ID' column.
        """

        snps = self.mutations[
            ~self.mutations["MUTATION"].str.contains(
                r"(?:indel|ins|del|Z|LOF)", regex=True
            )
        ].copy()

        if "SNP_ID" not in snps.columns:
            assert (
                "REF" in snps.columns and "ALT" in snps.columns
            ), "The DataFrame must contain either 'SNP_ID' or both 'REF' and 'ALT' columns."

            snps["SNP_ID"] = (
                snps["MUTATION"].apply(lambda i: i.split("@")[0]).astype(str)
                + "@"
                + snps["REF"].astype(str)
                + snps["MUTATION"].apply(lambda i: i.split("@")[1][1:-1]).astype(str)
                + snps["ALT"].astype(str)
            )

        return snps

    def calc_clusters(self, cluster_distance: int = 50) -> Sequence[int]:
        """
        Perform agglomerative clustering on SNP distances and map clusters back to all samples.

        Args:
            cluster_distance: SNP distance threshold for clustering.

        Returns:
            Series of cluster labels aligned to self.samples.UNIQUEID (0 indicates no SNP data).
        """
        snps = self.generate_snps_df()

        # Build sparse SNP matrix
        X_snps = self.build_X_sparse(snps)

        # Compute Hamming distances
        distances = self.hamming_distance(X_snps)

        # Perform agglomerative clustering
        agg_cluster = AgglomerativeClustering(
            metric="precomputed",
            linkage="complete",
            distance_threshold=cluster_distance,
            n_clusters=None,
        )

        # Fit clustering model and ensure starts from 1, not 0
        clusters = agg_cluster.fit_predict(distances)
        clusters += 1

        # Map clustering results back to all samples
        cluster_map = dict(zip(snps["UNIQUEID"].unique(), clusters))
        clusters = self.samples["UNIQUEID"].map(cluster_map).fillna(0).astype(int)

        return clusters.tolist()

    def define_intervals(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Define MIC intervals (low/high) under censoring and dilution rules, then log-transform.

        MIC encoding is expected as strings:
            - '<=x' left-censored
            - '>x' right-censored
            - 'x' exact

        Args:
            df: DataFrame containing a 'MIC' column.

        Returns:
            (y_low_log, y_high_log) arrays on the log(dilution_factor) scale.
        """

        y_low = np.zeros(len(df.MIC))
        y_high = np.zeros(len(df.MIC))

        if not self.censored:
            tail_dilution_factor = self.dilution_factor**self.tail_dilutions

        for i, mic in enumerate(df.MIC):
            if mic.startswith("<="):  # Left-censored
                lower_bound = float(mic[2:])
                y_low[i] = 1e-6 if self.censored else lower_bound / tail_dilution_factor
                y_high[i] = lower_bound
            elif mic.startswith(">"):  # Right-censored
                upper_bound = float(mic[1:])
                y_low[i] = upper_bound
                y_high[i] = (
                    np.inf if self.censored else upper_bound * tail_dilution_factor
                )
            else:  # Exact MIC value
                mic_value = float(mic)
                y_low[i] = mic_value / self.dilution_factor
                y_high[i] = mic_value

        # Apply log transformation to intervals
        return self.log_transf_intervals(y_low, y_high)

    def log_transf_intervals(
        self,
        y_low: np.ndarray,
        y_high: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply log transformation to interval bounds using log base = dilution_factor.
        """

        log_base = np.log(self.dilution_factor)

        # Initialize outputs with -inf (correct for log of non-positive lower bounds)
        y_low_log = np.full_like(y_low, -np.inf, dtype=float)
        y_high_log = np.full_like(y_high, -np.inf, dtype=float)

        # Compute logs only where valid
        np.log(y_low, where=(y_low > 0), out=y_low_log)
        np.log(y_high, where=(y_high > 0), out=y_high_log)

        y_low_log /= log_base
        y_high_log /= log_base

        return y_low_log, y_high_log


    def log_transf_val(self, val: float) -> float:
        """
        Log-transform a scalar value using log base = dilution_factor.

        Args:
            val: Positive scalar to transform.

        Returns:
            Log-transformed value.
        """

        log_base = np.log(self.dilution_factor)
        return float(np.log(val) / log_base)

    def initial_params(
        self,
        X: pd.DataFrame,
        y_low: np.ndarray,
        y_high: np.ndarray,
        clusters: Optional[Sequence[int]],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate initial parameters for the regression model.

        Strategy:
            - Use interval midpoints where finite.
            - Estimate beta via least squares on the finite subset.
            - Sample small random initial u (random effects).
            - Set sigma to log(std(midpoints)).

        Args:
            X: Binary design matrix.
            y_low: Lower interval bounds (log scale).
            y_high: Upper interval bounds (log scale).
            clusters: Cluster labels (or None).

        Returns:
            (beta_init, u_init, sigma_init) where sigma_init is on the log scale.
        """
        # Need to think about this a little more carefully - perhaps init params in meintreg could be improved?
        midpoints = (y_low + y_high) / 2.0
        valid_mask = np.isfinite(midpoints)
        X_valid = X[valid_mask]
        midpoints_valid = midpoints[valid_mask]
        # Initial estimate of beta via linear regression
        beta_init = np.linalg.lstsq(X_valid, midpoints_valid, rcond=None)[0]
        # Initial random effects - small non-zero value
        u_init = np.random.normal(loc=0, scale=0.1, size=len(np.unique(clusters or [])))
        # sigma - std of valid midpoints
        sigma = np.nanstd(midpoints_valid)
        sigma = np.log(sigma)

        return beta_init, u_init, sigma

    def fit(
        self,
        X: pd.DataFrame,
        y_low: np.ndarray,
        y_high: np.ndarray,
        random_effects: Optional[Sequence[int]] = None,
        bounds: Optional[list[tuple[Optional[float], Optional[float]]]] = None,
        options: Optional[dict[str, Any]] = None,
        L2_penalties: Optional[dict[str, Any]] = None,
    ) -> Any:
        """
        Fit the regression model to mutation and MIC interval data.

        Args:
            X: Binary design matrix.
            y_low: Lower interval bounds (log scale).
            y_high: Upper interval bounds (log scale).
            random_effects: Cluster labels or None if random effects are not used.
            bounds: Parameter bounds for optimization.
            options: Options passed to the optimizer.
            L2_penalties: Regularization settings for MeIntReg.

        Returns:
            Fitted MeIntReg result.
        """
        _b, _u, _s = self.initial_params(X, y_low, y_high, random_effects)

        if random_effects is not None:
            initial_params = np.concatenate([_b, _u, [_s]])
        else:
            initial_params = np.concatenate([_b, [_s]])

        if options:
            return MeIntReg(y_low, y_high, X.to_numpy(), random_effects).fit(
                method="L-BFGS-B",
                initial_params=initial_params,
                bounds=bounds,
                options=options,
                L2_penalties=L2_penalties,
            )
        else:
            return self.iter_tolerances(
                X, y_low, y_high, random_effects, initial_params, bounds, L2_penalties
            )

    def iter_tolerances(
        self,
        X: pd.DataFrame,
        y_low: np.ndarray,
        y_high: np.ndarray,
        clusters: Optional[Sequence[int]],
        initial_params: np.ndarray,
        bounds: Optional[list[tuple[Optional[float], Optional[float]]]],
        L2_penalties: Optional[dict[str, Any]] = None,
    ) -> Any:
        """
        Grid search over optimization tolerances to find a successful fit (early stops on success).

        Args:
            X: Binary design matrix.
            y_low: Lower interval bounds (log scale).
            y_high: Upper interval bounds (log scale).
            clusters: Cluster labels or None.
            initial_params: Initial optimization vector.
            bounds: Bounds for optimization parameters.
            L2_penalties: Regularization settings for MeIntReg.

        Returns:
            First successful optimization result; returns None if all attempts fail.
        """

        # may need to reduce maxfun search for speed up.
        # maxfun (number function evaluations) is generally too low
        # (default 15000) to fit, so can get a success either by
        # increasing or by loosening tolerances. Below tries to find a balance.

        maxiter = 10000
        maxfun = 50000
        gtols = [1e-5, 1e-4, 1e-3]
        ftols = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

        for gtol in gtols:
            for ftol in ftols:
                r = MeIntReg(y_low, y_high, X.to_numpy(), clusters).fit(
                    method="L-BFGS-B",
                    initial_params=initial_params,
                    bounds=bounds,
                    options={
                        "maxiter": maxiter,
                        "maxfun": maxfun,
                        "ftol": ftol,
                        "gtol": gtol,
                    },
                    L2_penalties=L2_penalties,
                )
                if r:
                    return r

    def predict_effects(
        self,
        b_bounds: tuple[Optional[float], Optional[float]] = (None, None),
        u_bounds: tuple[Optional[float], Optional[float]] = (None, None),
        s_bounds: tuple[Optional[float], Optional[float]] = (None, None),
        options: Optional[dict[str, Any]] = None,
        L2_penalties: Optional[dict[str, Any]] = None,
        fixed_effects: Optional[list[str]] = None,
        random_effects: bool = True,
        cluster_distance: int = 50,
    ) -> tuple[Any, pd.DataFrame]:
        """
        Fit the regression model and extract per-mutation effects.

        Args:
            b_bounds: Bounds for fixed effects coefficients (beta).
            u_bounds: Bounds for random effects coefficients (u).
            s_bounds: Bounds for standard deviation parameter (sigma, on log scale).
            options: Optimizer options.
            L2_penalties: Regularization settings.
            fixed_effects: Optional list of fixed-effect column names (must exist in samples df).
            random_effects: Whether to infer SNP clusters to model population structure.
            cluster_distance: SNP distance threshold for clustering.

        Returns:
            (model, effects) where effects is a DataFrame of mutation effect estimates.
        """

        validate_regression_predict_inputs(
            list(self.samples.columns),
            b_bounds,
            u_bounds,
            s_bounds,
            options,
            L2_penalties,
            fixed_effects,
            random_effects,
            cluster_distance,
            self.genes,
        )

        y_low, y_high = self.define_intervals(self.samples)

        if len(self.genes) > 0:
            self.target_mutations = self.mutations[
                self.mutations["MUTATION"].str.split("@").str[0].isin(self.genes)
            ]
        else:
            self.target_mutations = self.mutations

        self.df = pd.merge(
            self.samples, self.target_mutations, on=["UNIQUEID"], how="left"
        )

        X = self.build_X(self.df, fixed_effects=fixed_effects)

        if random_effects:
            clusters = self.calc_clusters(cluster_distance)
            u_bounds_ = [u_bounds] * len(np.unique(clusters))
        else:
            clusters = None
            u_bounds_ = []

        b_bounds_ = [b_bounds] * X.shape[1]
        bounds_ = b_bounds_ + u_bounds_ + [s_bounds]

        model = self.fit(X, y_low, y_high, clusters, bounds_, options, L2_penalties)

        effects = self.extract_effects(model, X, fixed_effects)

        return model, effects

    def extract_effects(
        self,
        model: Any,
        X: pd.DataFrame,
        fixed_effects: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Extract mutation effects from a fitted regression model and convert to MIC scale.

        If the fitted model exposes a Hessian inverse, standard errors are estimated and
        propagated to MIC scale.

        Args:
            model: Fitted MeIntReg result object.
            X: Design matrix used for fitting.
            fixed_effects: Optional list of fixed-effect field names (used to exclude one-hot FE columns).

        Returns:
            DataFrame with effect estimates:
                - Mutation
                - effect_size (log scale)
                - effect_std (optional)
                - MIC (original scale)
                - MIC_std (optional)
        """
        p = X.shape[1]

        fixed_effect_coefs = model.x[:p]

        columns_to_exclude = (
            {
                col
                for fe in fixed_effects
                for col in X.columns
                if col.startswith(f"{fe}_")
            }
            if fixed_effects
            else set()
        )

        # Filter out fixed-effect columns from the mutation columns
        mutation_columns = [col for col in X.columns if col not in columns_to_exclude]

        # Extract the corresponding coefficients
        mutation_effect_coefs = fixed_effect_coefs[
            [X.columns.get_loc(col) for col in mutation_columns]
        ]

        effects = pd.DataFrame(
            {
                "Mutation": mutation_columns,
                "effect_size": mutation_effect_coefs,
            }
        )
        # Convert effect sizes to MIC values (by reversing the log transformation)
        effects["MIC"] = self.dilution_factor ** effects["effect_size"]

        if hasattr(model, "hess_inv"):
            hess_inv_dense = model.hess_inv.todense()  # Convert to a dense matrix
            # Extract the diagonal elements corresponding to the fixed effects (log(MIC) scale)
            mutation_indices = [X.columns.get_loc(col) for col in mutation_columns]
            diag = np.diag(np.asarray(hess_inv_dense))
            idx = np.asarray(mutation_indices, dtype=np.intp)
            effect_std_log = np.sqrt(diag[idx])
            effects["effect_std"] = effect_std_log
            # Convert standard deviation to MIC scale
            effects["MIC_std"] = (
                effects["MIC"] * np.log(self.dilution_factor) * effects["effect_std"]
            )
            effects = effects[
                ["Mutation", "effect_size", "effect_std", "MIC", "MIC_std"]
            ]
        else:
            effects = effects[["Mutation", "effect_size", "MIC"]]

        return effects

    @staticmethod
    def z_test(mu: float, val: float, se: float) -> Any:
        """
        Compute a two-tailed z-test p-value.

        Args:
            mu: Observed/estimated mean.
            val: Null/reference value.
            se: Standard error.

        Returns:
            Two-tailed p-value.
        """
        z = (mu - val) / se
        p_value = 2 * (1 - norm.cdf(abs(z)))
        return p_value

    def classify_effects(
        self,
        effects: pd.DataFrame,
        ecoff: float,
        p: float = 0.95,
    ) -> tuple[pd.DataFrame, float]:
        """
        Classify mutation effects as Resistant (R), Susceptible (S), or Undetermined (U) using a z-test.

        Effects are classified by comparing effect_size to the (log-space) breakpoint and applying
        a two-tailed z-test using effect_std.

        Args:
            effects: Effects DataFrame with 'effect_size' and 'effect_std'.
            p: Confidence parameter (default 0.95).

        Returns:
            (effects, ecoff) where effects includes 'p_value' and 'Classification'.
        """

        validate_regression_classify_inputs(ecoff, p)

        breakpoint = self.log_transf_val(ecoff)

        effects["p_value"] = effects.apply(
            lambda row: self.z_test(row["effect_size"], breakpoint, row["effect_std"]),
            axis=1,
        )

        effects["Classification"] = np.select(
            condlist=[
                (effects["effect_size"] > breakpoint) & (effects["p_value"] < (1 - p)),
                (effects["effect_size"] < breakpoint) & (effects["p_value"] < (1 - p)),
            ],
            choicelist=["R", "S"],
            default="U",
        )

        return effects, ecoff

    def add_mutation(
        self, mutation: str, prediction: str, evidence: dict[str, Any]
    ) -> None:
        """
        Add a mutation entry to the catalogue and record insertion order.

        Args:
            mutation: Mutation identifier.
            prediction: Phenotype label ('R', 'S', or 'U').
            evidence: Evidence metadata for the entry.

        Returns:
            None
        """
        self.catalogue[mutation] = {"pred": prediction, "evid": evidence}
        self.entry.append(mutation)

    def build(
        self,
        ecoff: float,
        b_bounds: tuple[Optional[float], Optional[float]] = (None, None),
        u_bounds: tuple[Optional[float], Optional[float]] = (None, None),
        s_bounds: tuple[Optional[float], Optional[float]] = (None, None),
        options: Optional[dict[str, Any]] = None,
        L2_penalties: Optional[dict[str, Any]] = None,
        p: float = 0.95,
        fixed_effects: Optional[list[str]] = None,
        random_effects: bool = True,
        cluster_distance: int = 50,
    ) -> "RegressionBuilder":
        """
        Orchestrate model fitting, effect extraction, classification, and catalogue construction.

        Args:
            b_bounds: Bounds for fixed effects coefficients (beta).
            u_bounds: Bounds for random effects coefficients (u).
            s_bounds: Bounds for standard deviation parameter (sigma, log scale).
            options: Optimizer options; if None/empty, an internal tolerance grid search is used.
            L2_penalties: Regularization settings passed to the fitter.
            ecoff: ECOFF on MIC scale.
            p: Confidence parameter (default 0.95).
            fixed_effects: Optional list of fixed-effect columns in samples df.
            random_effects: Whether to model population structure using SNP clusters.
            cluster_distance: SNP distance threshold used for clustering (if enabled).

        Returns:
            self: The built RegressionBuilder instance.
        """
        # Predict effects
        _, effects = self.predict_effects(
            b_bounds=b_bounds,
            u_bounds=u_bounds,
            s_bounds=s_bounds,
            options=options,
            L2_penalties=L2_penalties,
            fixed_effects=fixed_effects,
            random_effects=random_effects,
            cluster_distance=cluster_distance,
        )

        effects, ecoff = self.classify_effects(
            effects, ecoff=ecoff, p=p
        )

        breakpoint = self.log_transf_val(ecoff)

        def add_mutation_from_row(row: pd.Series) -> None:
            evidence: dict[str, Any] = {
                "MIC": row.get("MIC"),
                "ECOFF": ecoff,
                "effect_size": row.get("effect_size"),
                "breakpoint": breakpoint,
                "p_value": row.get("p_value"),
            }
            # Only attach std fields if present.
            if "MIC_std" in row:
                evidence["MIC_std"] = row.get("MIC_std")
            if "effect_std" in row:
                evidence["effect_std"] = row.get("effect_std")

            self.add_mutation(str(row["Mutation"]), str(row["Classification"]), evidence)

        for _, row in effects.iterrows():
            add_mutation_from_row(row)

        return self

    def return_catalogue(self) -> dict[str, dict[str, Any]]:
        """
        Return the catalogue ordered by insertion.

        Returns:
            Ordered catalogue mapping mutation -> {'pred': ..., 'evid': ...}.
        """

        return {key: self.catalogue[key] for key in self.entry if key in self.catalogue}

    def to_json(self, outfile: str) -> None:
        """
        Export the catalogue to a JSON file.

        Args:
            outfile: Path to output JSON file.

        Returns:
            None
        """
        with open(outfile, "w") as f:
            json.dump(self.catalogue, f, indent=4)
