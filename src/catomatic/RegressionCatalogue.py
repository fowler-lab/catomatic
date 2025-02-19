import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.stats import norm
from .Ecoff import EcoffGenerator
from .PiezoTools import PiezoExporter
from .defence_module import (
    validate_regression_init,
    validate_regression_predict_inputs,
    validate_regression_classify_inputs,
)
from intreg.meintreg import MeIntReg
from sklearn.cluster import AgglomerativeClustering


class RegressionBuilder(PiezoExporter):
    """
    Builds a mutation catalogue compatible with Piezo in a standardized format.

    MICs are treated as intervals to fit a regression curve assuming a Gaussian distribution.
    """

    def __init__(
        self,
        samples,
        mutations,
        genes=[],
        dilution_factor=2,
        censored=True,
        tail_dilutions=1,
        FRS=None,
        seed=0,
    ):
        """
        Initialize the ECOFF generator with sample and mutation data.

        Args:
            samples (DataFrame): DataFrame containing 'UNIQUEID' and 'MIC' columns.
            mutations (DataFrame): DataFrame containing 'UNIQUEID' and 'MUTATION' columns.
            genes (list, optional): A list of RAV genes. A list must be supplied if non-RAV
                genes are in the mutations table (ie if clustering snp distances)
            dilution_factor (int): The factor for dilution scaling (default is 2 for doubling).
            censored (bool): Flag to indicate if censored data is used.
            tail_dilutions (int): Number of dilutions to extend for interval tails if uncensored.
            FRS: Fraction of read support to filter mutations by (default None).
            seed: Numpy random seed (only pertains to initial parameter generator)
        """

        samples = pd.read_csv(samples) if isinstance(samples, str) else samples
        mutations = pd.read_csv(mutations) if isinstance(mutations, str) else mutations

        validate_regression_init(
            samples,
            mutations,
            genes,
            dilution_factor,
            censored,
            tail_dilutions,
            FRS,
            seed,
        )

        if FRS is not None:
            # note this will filter out mutations for clustering as well
            mutations = mutations[mutations.FRS >= FRS]

        self.samples, self.mutations = samples, mutations

        self.genes = genes
        self.dilution_factor = dilution_factor
        self.censored = censored
        self.tail_dilutions = tail_dilutions
        np.random.seed(seed)

        # instantiate catalogue object
        self.catalogue = {}
        self.entry = []

    def build_X(self, df, fixed_effects=None):
        """
        Build a binary mutation matrix X and optionally include fixed effects.

        Args:
            df (DataFrame): DataFrame containing mutation data and optionally additional fixed effect columns.
            fixed_effects (list of str, optional): List of column names in `df` to include as fixed effects. Defaults to None

        Returns:
            DataFrame: Binary mutation matrix with optional fixed effects appended as additional columns.
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
    def build_X_sparse(df):
        """
        Build a sparse binary mutation matrix.

        Args:
            df (DataFrame): DataFrame containing sample identifiers ('UNIQUEID') and
                            mutation identifiers ('SNP_ID').

        Returns:
            csr_matrix: Sparse binary matrix where rows represent unique samples and
                        columns represent unique mutations
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
    def hamming_distance(X_sparse, n_jobs=-1, block_size=1000):
        """
        Compute pairwise absolute Hamming distance for a sparse binary matrix.

        Args:
            X_sparse (csr_matrix): Sparse binary mutation matrix.
            n_jobs (int): Number of parallel jobs (-1 uses all available cores).
            block_size (int): Size of blocks for chunked computation.

        Returns:
            ndarray: Pairwise absolute Hamming distance matrix.
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

    def generate_snps_df(self):
        """
        Generate a filtered SNP DataFrame, ensuring a snp_id columns

        Returns:
            DataFrame: A filtered and processed DataFrame of SNPs.
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

    def calc_clusters(self, cluster_distance=50):
        """
        Perform agglomerative clustering on a SNP matrix and map clusters back to samples.

        Args:
            cluster_distance (int): SNP distance threshold for clustering.

        Returns:
            ndarray: Cluster labels for all samples in self.samples, ordered by self.samples.UNIQUEID.
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

        return clusters

    def define_intervals(self, df):
        """
        Define MIC intervals based on the dilution factor and censoring settings.

        Args:
            df (DataFrame): DataFrame containing MIC data.

        Returns:
            tuple: Log-transformed lower and upper bounds for MIC intervals.
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

    def log_transf_intervals(self, y_low, y_high):
        """
        Apply log transformation to interval bounds with the specified dilution factor.

        Args:
            y_low (array-like): Lower bounds of the intervals.
            y_high (array-like): Upper bounds of the intervals.

        Returns:
            tuple: Log-transformed lower and upper bounds.
        """
        log_base = np.log(self.dilution_factor)

        y_low_log = np.log(y_low, where=(y_low > 0)) / log_base
        y_high_log = np.log(y_high, where=(y_high > 0)) / log_base

        return y_low_log, y_high_log

    def log_transf_val(self, val):
        """
        Calculate the logarithm of a value using the dilution factor as the base.

        Args:
            val (float): The value to be log-transformed. Must be positive.

        Returns:
            float: The log-transformed value in the specified base (dilution factor).
        """

        log_base = np.log(self.dilution_factor)
        return np.log(val) / log_base

    def initial_params(self, X, y_low, y_high, clusters):
        """
        Generate initial parameters for the regression model.

        Args:
            X (DataFrame): Binary mutation matrix.
            y_low (array-like): Lower MIC bounds.
            y_high (array-like): Upper MIC bounds.
            clusters (array-like): Cluster labels for samples.

        Returns:
            tuple: Initial beta, u (cluster effects), and sigma parameters.
        """
        # Need to think about this a little more carefully - perhaps init params in meintreg could be improved?
        midpoints = (y_low + y_high) / 2.0
        valid_mask = np.isfinite(midpoints)
        X_valid = X[valid_mask]
        midpoints_valid = midpoints[valid_mask]
        # Initial estimate of beta via linear regression
        beta_init = np.linalg.lstsq(X_valid, midpoints_valid, rcond=None)[0]
        # Initial random effects - small non-zero value
        u_init = np.random.normal(loc=0, scale=0.1, size=len(np.unique(clusters)))
        # sigma - std of valid midpoints
        sigma = np.nanstd(midpoints_valid)
        sigma = np.log(sigma)

        return beta_init, u_init, sigma

    def fit(
        self,
        X,
        y_low,
        y_high,
        random_effects=None,
        bounds=None,
        options={},
        L2_penalties={},
    ):
        """
        Fit the regression model to the mutation and MIC interval data.

        Args:
            X (DataFrame): Binary mutation matrix.
            y_low (array-like): Lower MIC bounds.
            y_high (array-like): Upper MIC bounds.
            random_effects (array-like or None): Cluster labels or None if random effects are not used.
            bounds: Parameter bounds.
            options (dict): Options for optimization.
            L2_penalties (dict): Regularization penalties.

        Returns:
            MeIntReg: Fitted regression model.
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
        self, X, y_low, y_high, clusters, initial_params, bounds, L2_penalties
    ):
        """
        Perform a grid search over optimization tolerances to find a successful fit, with
        early stopping on succes.

        Args:
            X (DataFrame): Binary mutation matrix.
            y_low (array-like): Lower MIC bounds.
            y_high (array-like): Upper MIC bounds.
            clusters (array-like): Cluster labels for each sample.
            initial_params (array-like): Initial parameter guesses for optimization.
            bounds (list): Bounds for optimization parameters.

        Returns:
            OptimizeResult: The first successful fit result.
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
                if r.success:
                    return r

    def predict_effects(
        self,
        b_bounds=(None, None),
        u_bounds=(None, None),
        s_bounds=(None, None),
        options=None,
        L2_penalties=None,
        fixed_effects=None,
        random_effects=True,
        cluster_distance=50,
    ):
        """
        Predict mutation effects using the fitted regression model.

        Args:
            b_bounds (tuple or None): Bounds for the fixed effects coefficients (beta).
            u_bounds (tuple or None): Bounds for the random effects (u).
            s_bounds (tuple or None): Bounds for the standard deviation parameter (sigma).
            options (dict or None): Options for scipy minimize.
            L2_penalties (dict or None): Regularization strengths for fixed and random effects.
            fixed_effects (list of str, optional): List of fixed effect column names - must exist in samples df. Defaults to None
            random_effects (bool): Whether to calculate SNP clusters for population structure.
            cluster_distance (int): Distance threshold for clustering.

        Returns:
            tuple: Fitted regression model and mutation matrix X.
        """

        validate_regression_predict_inputs(
            self.samples.columns,
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
            u_bounds = [u_bounds] * len(np.unique(clusters))
        else:
            clusters = None
            u_bounds = []

        b_bounds = [b_bounds] * X.shape[1]
        bounds = b_bounds + u_bounds + [s_bounds]

        model = self.fit(X, y_low, y_high, clusters, bounds, options, L2_penalties)

        effects = self.extract_effects(model, X, fixed_effects)

        return model, effects

    def extract_effects(self, model, X, fixed_effects=None):
        """
        Extract mutation effects from a fitted regression model and calculate their MIC values.

        Args:
            model (MeIntReg): The fitted regression model, which contains fixed-effect coefficients
                and possibly a Hessian inverse matrix for uncertainty estimation.
            X (DataFrame): Binary mutation matrix with mutations and possibly fixed effects as columns.
            fixed_effects (list of str, optional): List of fixed effect column names. Defaults to None.

        Returns:
            DataFrame: A DataFrame with the following columns:
                - "Mutation": Names of the mutations.
                - "effect_size": The effect size (log-transformed scale).
                - "effect_std" (optional): The standard deviation of the effect size (log scale),
                if available from the model.
                - "MIC": The Minimum Inhibitory Concentration (MIC) calculated by reversing the
                log transformation.
                - "MIC_std" (optional): The standard deviation of the MIC, if available.
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
            effect_std_log = np.sqrt(np.diag(hess_inv_dense)[mutation_indices])
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
    def z_test(mu, val, se):
        """
        Perform a z-test to calculate the two-tailed p-value.

        Args:
            mu (float): The mean value (e.g., observed or estimated mean).
            val (float): The value to compare against (e.g., hypothesized mean).
            se (float): The standard error of the mean.

        Returns:
            float: The p-value for the two-tailed z-test.
        """

        z = (mu - val) / se
        p_value = 2 * (1 - norm.cdf(abs(z)))
        return p_value

    def classify_effects(self, effects, ecoff=None, percentile=99, p=0.95):
        """Classify mutation effects as Resistant (R), Susceptible (S), or Undetermined (U) using a Z-test.

        Args:
            effects (DataFrame): A DataFrame containing mutation effects with columns
                'effect_size' and 'effect_std'.
            ecoff (float, optional): The epidemiological cutoff (ECOFF) value. If None, it will
                be calculated using the GenerateEcoff method.
            percentile (int, optional): Percentile used to calculate the ECOFF if ecoff is None
                (default is 99).
            p (float, optional): Significance level for statistical testing (default is 0.95).

        Returns:
            tuple: A tuple containing:
                - effects (DataFrame): Updated DataFrame with new 'p_value' and 'Classification' columns.
                - ecoff (float): The ECOFF value used for classification."""

        validate_regression_classify_inputs(ecoff, percentile, p)

        if ecoff is None:
            ecoff, breakpoint, _, _, _ = EcoffGenerator(
                self.samples,
                self.target_mutations,
                dilution_factor=self.dilution_factor,
                censored=self.censored,
                tail_dilutions=self.tail_dilutions,
            ).generate(percentile)
        else:
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

    def add_mutation(self, mutation, prediction, evidence):
        """
        Adds mutation to cataloue object, and indexes to track order.

        Parameters:
            mutation (str): mutaiton to be added
            prediction (str): phenotype of mutation
            evidence (any): additional metadata to be added
        """

        self.catalogue[mutation] = {"pred": prediction, "evid": evidence}
        # record entry once mutation is added
        self.entry.append(mutation)

    def build(
        self,
        b_bounds=(None, None),
        u_bounds=(None, None),
        s_bounds=(None, None),
        options=None,
        L2_penalties=None,
        ecoff=None,
        percentile=99,
        p=0.95,
        fixed_effects=None,
        random_effects=True,
        cluster_distance=50,
    ):
        """
        Constructs a mutation catalogue by predicting mutation effects and classifying them as resistant, susceptible, or undetermined.
        Uses regression modeling to estimate the effects of mutations on observed MIC values. It classifies mutations based
        on statistical tests and applies ECOFF thresholds to determine phenotype categories. The results are stored in the catalogue.

        Args:
            b_bounds (tuple, optional): Bounds for fixed effects coefficients (min, max). Defaults to (None, None).
            u_bounds (tuple, optional): Bounds for random effects coefficients (min, max). Defaults to (None, None).
            s_bounds (tuple, optional): Bounds for the standard deviation parameter (min, max). Defaults to (None, None).
            options (dict, optional): Scipy minimise's ptimization options for the regression fitting. Defaults to None.
            L2_penalties (dict, optional): Regularization penalties for fixed and random effects. Defaults to None.
            ecoff (float, optional): Epidemiological cutoff value for classification, in logspace. If None, it will be calculated. Defaults to None.
            percentile (int/float, optional): Percentile for ECOFF calculation if ecoff is None. Defaults to 99.
            p (float, optional): Significance level for classification. Defaults to 0.95.
            fixed_effects (list of str, optional): List of fixed effect column names - column must exist in the samples df. Defaults to None
            random_effects (bool): Whether to calculate and include random effects (snp distance clusters)
            cluster_distance (float): v
        Returns:
            RegressionBuilder: The instance with the updated mutation catalogue.
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
            effects, ecoff=ecoff, percentile=percentile, p=p
        )

        def add_mutation_from_row(row):
            evidence = {
                "MIC": row["MIC"],
                "MIC_std": row["MIC_std"],
                "ECOFF": ecoff,
                "effect_size": row["effect_size"],
                "effect_std": row["effect_std"],
                "breakpoint": self.log_transf_val(ecoff),
                "p_value": row["p_value"],
            }
            self.add_mutation(row["Mutation"], row["Classification"], evidence)

        effects.apply(add_mutation_from_row, axis=1)

        return self

    def return_catalogue(self):
        """
        Public method that returns the catalogue dictionary.

        Returns:
            dict: The catalogue data stored in the instance.
        """

        return {key: self.catalogue[key] for key in self.entry if key in self.catalogue}

    def to_json(self, outfile):
        """
        Exports the catalogue to a JSON file.

        Parameters:
            outfile (str): The path to the output JSON file where the catalogue will be saved.
        """
        with open(outfile, "w") as f:
            json.dump(self.catalogue, f, indent=4)

