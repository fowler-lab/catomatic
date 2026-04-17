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
    Build a mutation-level MIC catalogue using mixed-effects interval regression.

    MICs are treated as interval-censored measurements on a log(dilution_factor) scale.
    A Gaussian mixed-effects model is fitted:

        log2(MIC*) = β0 + Xβ + u_cluster + ε

    where:
        β0        = baseline (intercept) log2 MIC
        Xβ        = mutation and optional fixed-effect contributions
        u_cluster = population-structure random intercept (optional)
        ε         = residual Gaussian noise (shared σ)

    Mutation effects are interpreted as log2 shifts relative to the baseline,
    and are converted back to absolute MIC scale using:

        MIC_mutation = dilution_factor^(β0 + β_mutation)

    Standard errors are propagated using the full covariance structure
    (including intercept–mutation covariance).

    `build()` orchestrates fitting, effect extraction, classification relative
    to an ECOFF, and catalogue construction.

    Args:
        samples : pd.DataFrame | str
            Table (or CSV path) with columns ['UNIQUEID', 'MIC'].
        mutations : pd.DataFrame | str
            Table (or CSV path) with columns ['UNIQUEID', 'MUTATION'].
            Optional columns: ['FRS', 'REF', 'ALT', 'SNP_ID'].
        genes : list[str], optional
            Restrict modelling to mutations in these genes.
        dilution_factor : int, default=2
            MIC dilution base.
        censored : bool, default=True
            Whether to treat interval tails as censored.
        tail_dilutions : int, default=1
            Number of dilutions used to extend tails when `censored=False`.
        frs : float, optional
            Fraction read support threshold for filtering mutations.
        seed : int, default=0
            Random seed for initial parameter generation.
    """

    samples: pd.DataFrame
    mutations: pd.DataFrame
    catalogue: dict[str, dict[str, Any]]
    entry: list[str]

    genes: list[str]
    dilution_factor: int
    censored: bool
    tail_dilutions: int

    clusters: Optional[Sequence[int]]
    X: Optional[pd.DataFrame]

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
        Construct the fixed-effect design matrix.

        Creates a binary mutation matrix (one column per mutation) and optionally
        appends one-hot encoded fixed effects (e.g. subspecies). An explicit
        intercept column is always inserted as the first column.

        Args
        ----------
        df : pd.DataFrame
            Must contain ['UNIQUEID', 'MUTATION'] and any requested fixed-effect columns.

        fixed_effects : list[str], optional
            Column names in `df` to include as fixed effects.

        Returns
        -------
        pd.DataFrame
            Design matrix indexed by UNIQUEID with columns:

                Intercept
                Mutation indicators
                Optional fixed-effect indicators
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

        #add intercept column
        X.insert(0, "Intercept", 1)

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
        Infer population clusters from SNP Hamming distances.

        Constructs a SNP presence/absence matrix, computes pairwise Hamming
        distances, and performs agglomerative clustering using a complete
        linkage strategy with a specified distance threshold.

        Args:
            cluster_distance : int SNP distance threshold for clustering.

        Returns:
            list[int] Cluster labels aligned to `self.samples`.
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

        # Map clustering results back to all samples
        cluster_map = dict(zip(snps["UNIQUEID"].unique(), clusters))
        clusters = self.samples["UNIQUEID"].map(cluster_map)

        # Assign NaNs to a new cluster index
        if clusters.isna().any():
            max_label = clusters.max()
            clusters = clusters.fillna(max_label + 1)

        clusters = clusters.astype(int).tolist()
        return clusters

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
        Generate initial Args for the regression model.

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
        Fit the mixed-effects interval regression model.

        Initial Args are generated via least-squares on interval midpoints.
        The model is then fitted using L-BFGS-B optimization.

        Args:
            X : pd.DataFrame
                Fixed-effect design matrix including intercept.
            y_low : np.ndarray
                Lower interval bounds (log scale).
            y_high : np.ndarray
                Upper interval bounds (log scale).
            random_effects : sequence[int], optional
                Cluster labels defining random intercept groups.
            bounds : list[tuple], optional
                Parameter bounds applied in order to:
                    β (fixed effects), u (random intercepts), log(σ).
            options : dict, optional
                Optimizer settings passed to scipy.optimize.minimize.
            L2_penalties : dict, optional
                Ridge penalties with keys: 'lambda_beta', 'lambda_u', 'lambda_sigma'.

        Returns:
            MeIntReg
                Fitted model instance containing optimization results.
        """
        _b, _u, _s = self.initial_params(X, y_low, y_high, random_effects)

        if random_effects is not None:
            initial_params = np.concatenate([_b, _u, [_s]])
        else:
            initial_params = np.concatenate([_b, [_s]])


        return MeIntReg(y_low, y_high, X.to_numpy(), random_effects).fit(
            method="L-BFGS-B",
            initial_params=initial_params,
            bounds=bounds,
            options=options,
            L2_penalties=L2_penalties,
        )


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
        Fit the regression model and extract mutation-level effect estimates.

        1. Defines MIC intervals.
        2. Builds the design matrix (including intercept and optional fixed effects).
        3. Optionally infers population clusters for random intercepts.
        4. Fits the mixed-effects interval regression.
        5. Extracts per-mutation effect sizes and uncertainties.

        The design matrix (`self.X`) and cluster assignments (`self.clusters`)
        are stored for post-fit inspection.

        Args:
            b_bounds, u_bounds, s_bounds : tuple
                Bounds for fixed effects, random effects, and log(σ).
            options : dict, optional
                Optimizer settings.
            L2_penalties : dict, optional
                Ridge regularization Args.
            fixed_effects : list[str], optional
                Additional fixed-effect columns to include.
            random_effects : bool, default=True
                hether to include lineage random intercepts.
            cluster_distance : int, default=50
                SNP distance threshold for clustering.

        Returns:
            (model, effects) : tuple
                model : Fitted MeIntReg object.
                effects : pd.DataFrame with mutation effect estimates.
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

        #don't fit synonymous mutations (theyre used for clustering)
        aa = self.mutations["MUTATION"].str.split("@").str[-1]
        syn_mask = aa.str.match(r'^[A-Z].*[A-Z]$') & (aa.str[0] == aa.str[-1])

        if len(self.genes) > 0:
            self.target_mutations = self.mutations[
                self.mutations["MUTATION"].str.split("@").str[0].isin(self.genes)
                & (~syn_mask)
            ]
        else:
            self.target_mutations = self.mutations[~syn_mask]

        self.df = pd.merge(
            self.samples, self.target_mutations, on=["UNIQUEID"], how="left"
        )

        X = self.build_X(self.df, fixed_effects=fixed_effects)
        self.X = X

        if random_effects:
            self.clusters = self.calc_clusters(cluster_distance)
            u_bounds_ = [u_bounds] * len(np.unique(self.clusters))
        else:
            self.clusters = None
            u_bounds_ = []

        b_bounds_ = [b_bounds] * X.shape[1]
        bounds_ = b_bounds_ + u_bounds_ + [s_bounds]

        model = self.fit(X, y_low, y_high, self.clusters, bounds_, options, L2_penalties)

        print (model.result)

        effects = self.extract_effects(model, X, fixed_effects)

        return model, effects

    def extract_effects(
        self,
        model: Any,
        X: pd.DataFrame,
        fixed_effects: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Extract mutation-level effects from a fitted model.

        For each mutation, the estimated effect represents the log2 shift
        relative to the baseline intercept:

            effect_size = β_mutation

        Absolute predicted MIC is computed as:

            MIC = dilution_factor^(β0 + β_mutation)

        Standard errors are derived from the inverse Hessian and include
        covariance between the intercept and mutation coefficient:

            Var(β0 + β_mutation)

        Uncertainty is propagated to MIC scale via the delta method.

        Args:
            model : MeIntReg
                Fitted regression model.
            X : pd.DataFrame
                Design matrix used for fitting.
            fixed_effects : list[str], optional
                Names of fixed-effect columns to exclude from mutation extraction.

        Returns:
            pd.DataFrame
                Columns:
                    Mutation
                    effect_size (log2 shift)
                    effect_std (log2 scale, if available)
                    fold_change (relative to baseline)
                    MIC (absolute predicted MIC)
                    MIC_std (delta-method SE, if available)
        """
        p = X.shape[1]
        coefs = model.result.x[:p]

        # Identify intercept
        intercept_idx = X.columns.get_loc("Intercept")
        beta0 = coefs[intercept_idx]

        # Identify fixed-effect dummy columns (to exclude from mutation list)
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

        columns_to_exclude.add("Intercept")

        mutation_columns = [
            col for col in X.columns if col not in columns_to_exclude
        ]

        mutation_indices = [
            X.columns.get_loc(col) for col in mutation_columns
        ]

        mutation_effects = coefs[mutation_indices]

        effects = pd.DataFrame(
            {
                "Mutation": mutation_columns,
                "effect_size": mutation_effects,  # log2 shift
            }
        )

        # Baseline MIC
        baseline_MIC = self.dilution_factor ** beta0
        # Fold change from mutation
        effects["fold_change"] = (
            self.dilution_factor ** effects["effect_size"]
        )
        # Absolute predicted MIC
        effects["MIC"] = baseline_MIC * effects["fold_change"]

        # If Hessian available, compute SEs properly
        if hasattr(model.result, "hess_inv"):

            # Convert L-BFGS product to dense matrix
            hess_inv_dense = np.asarray(model.result.hess_inv.todense())

            beta0_idx = intercept_idx

            theta_stds = []
            mic_stds = []

            for m_idx in mutation_indices:

                # Variance of (beta0 + beta_m)
                var_theta = (
                    hess_inv_dense[beta0_idx, beta0_idx]
                    + hess_inv_dense[m_idx, m_idx]
                    + 2 * hess_inv_dense[beta0_idx, m_idx]
                )

                # Numerical guard
                var_theta = max(var_theta, 0.0)

                sd_theta = np.sqrt(var_theta)
                theta_stds.append(sd_theta)

                # Delta method to MIC scale
                mic_sd = (
                    effects.loc[
                        effects["Mutation"] == X.columns[m_idx], "MIC"
                    ].values[0]
                    * np.log(self.dilution_factor)
                    * sd_theta
                )

                mic_stds.append(mic_sd)

            effects["effect_std"] = theta_stds
            effects["MIC_std"] = mic_stds

            effects = effects[
                [
                    "Mutation",
                    "effect_size",
                    "effect_std",
                    "fold_change",
                    "MIC",
                    "MIC_std",
                ]
            ]
        else:
            effects = effects[
                ["Mutation", "effect_size", "fold_change", "MIC"]
            ]

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
        Run full catalogue construction workflow.

        Steps:
            1. Fit mixed-effects interval regression.
            2. Extract mutation effects.
            3. Classify effects relative to ECOFF using z-tests.
            4. Construct ordered mutation catalogue.

        Args:
            ecoff : float
                ECOFF on MIC scale.
            b_bounds, u_bounds, s_bounds : tuple
                Parameter bounds.
            options : dict, optional
                Optimizer settings.
            L2_penalties : dict, optional
                Ridge regularization.
            p : float, default=0.95
                Confidence level for classification.
            fixed_effects : list[str], optional
                Additional fixed-effect columns.
            random_effects : bool, default=True
                Whether to include lineage random intercepts.
            cluster_distance : int
                SNP clustering threshold.

        Returns:
            RegressionBuilder
                Fitted builder with populated catalogue.
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
