import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import norm
from .Ecoff import EcoffGenerator
from .PiezoTools import PiezoExporter
from .cli_module import main_regression_builder
from intreg.meintreg import MeIntReg
from sklearn.metrics import pairwise_distances
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
        dilution_factor=2,
        censored=True,
        tail_dilutions=1,
        cluster_distance=1,
        FRS=None,
    ):
        """
        Initialize the ECOFF generator with sample and mutation data.

        Args:
            samples (DataFrame): DataFrame containing 'UNIQUEID' and 'MIC' columns.
            mutations (DataFrame): DataFrame containing 'UNIQUEID' and 'MUTATION' columns.
            dilution_factor (int): The factor for dilution scaling (default is 2 for doubling).
            censored (bool): Flag to indicate if censored data is used.
            tail_dilutions (int): Number of dilutions to extend for interval tails if uncensored.
            cluster_distance (float): Distance threshold for clustering.
            FRS: Placeholder for future functionality (default None).
        """

        samples = pd.read_csv(samples) if isinstance(samples, str) else samples
        mutations = pd.read_csv(mutations) if isinstance(mutations, str) else mutations

        self.samples, self.mutations = samples, mutations

        self.df = pd.merge(samples, mutations, how="left", on=["UNIQUEID"])

        self.dilution_factor = dilution_factor
        self.censored = censored
        self.tail_dilutions = tail_dilutions
        self.cluster_distance = cluster_distance

        # instantiate catalogue object
        self.catalogue = {}
        self.entry = []

    def build_X(self, df):
        """
        Build the binary mutation matrix X.

        Args:
            df (DataFrame): DataFrame containing mutation data.

        Returns:
            DataFrame: Binary mutation matrix (1 for presence, 0 for absence).
        """
        ids = df.UNIQUEID.unique()

        # Create the pivot table and directly apply binary transformation
        X = pd.pivot_table(
            df,
            index="UNIQUEID",
            columns="MUTATION",
            aggfunc=lambda x: 1,  # Directly map presence to 1
            fill_value=0,  # Absence is 0
        )

        # Reindex to include all IDs, even those without mutations
        X = X.reindex(ids, fill_value=0)

        return X

    def cluster_coords(self, X):
        """
        Perform agglomerative clustering on the mutation matrix.

        Args:
            X (DataFrame): Binary mutation matrix.

        Returns:
            ndarray: Cluster labels for each sample.
        """
        # Josh ran on whole genome matrix, not candidate gene matrix
        dist_matrix = pairwise_distances(X, metric="hamming")

        agg_cluster = AgglomerativeClustering(
            metric="precomputed",
            linkage="complete",
            distance_threshold=self.cluster_distance
            / len(X.columns),  # Hamming threshold conversion
            n_clusters=None,
        )

        clusters = agg_cluster.fit_predict(dist_matrix)

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
        p = X.shape[1]
        midpoints = (y_low + y_high) / 2.0
        valid_midpoints = np.where(np.isfinite(midpoints), midpoints, np.nan)
        valid_indices = ~np.isnan(valid_midpoints)
        X_valid = X[valid_indices]
        midpoints_valid = valid_midpoints[valid_indices]
        # Initial estimate of beta via linear regression
        beta_init = np.linalg.lstsq(X_valid, midpoints_valid, rcond=None)[0]
        # Initial random effects - small non-zero value
        u_init = np.random.normal(loc=0, scale=0.1, size=len(np.unique(clusters)))
        # sigma = 1  # Fixed initial sigma
        sigma = np.random.uniform(0.5, 2.0)

        return beta_init, u_init, sigma

    def fit(self, X, y_low, y_high, clusters, bounds=None, options={}, L2_penalties={}):
        """
        Fit the regression model to the mutation and MIC interval data.

        Args:
            X (DataFrame): Binary mutation matrix.
            y_low (array-like): Lower MIC bounds.
            y_high (array-like): Upper MIC bounds.
            clusters (array-like): Cluster labels.
            bounds: parameter bounds
            options (dict or None): options for scipy minimise

        Returns:
            MeIntReg: Fitted regression model.
        """
        _b, _u, _s = self.initial_params(X, y_low, y_high, clusters)
        initial_params = np.concatenate([_b, _u, [_s]])

        if options is not None and len(options) > 0:
            return MeIntReg(y_low, y_high, X, clusters).fit(
                method="L-BFGS-B",
                initial_params=initial_params,
                bounds=bounds,
                options=options,
                L2_penalties=L2_penalties,
            )
        else:
            return self.iter_tolerances(
                X, y_low, y_high, clusters, initial_params, bounds, L2_penalties
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
                r = MeIntReg(y_low, y_high, X, clusters).fit(
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
    ):
        """
        Predict mutation effects using the fitted regression model.

        Args:
            b_bounds (tuple or None): Bounds for the fixed effects coefficients (\(\beta\)) as a
                tuple of (min, max). Use (None, None) for no bounds (default).
            u_bounds (tuple or None): Bounds for the random effects (\(u\)) as a tuple of (min, max).
                Use (None, None) for no bounds (default).
            s_bounds (tuple or None): Bounds for the standard deviation parameter (\(\sigma\)) as a
                tuple of (min, max). Use (None, None) for no bounds (default).
            options (dict or None): options for scipy minimise (check scipy docs)
            L2_penalties (dict or None): Regularisation strengths for fixed and random effects {lambda_beta:..., lambda_u:...}

        Returns:
            tuple: Fitted regression model and mutation matrix X.
        """
        y_low, y_high = self.define_intervals(self.samples)
        X = self.build_X(self.df)
        
        clusters = self.cluster_coords(X)

        b_bounds = [b_bounds] * X.shape[1]
        u_bounds = [u_bounds] * len(np.unique(clusters))
        bounds = b_bounds + u_bounds + [s_bounds]

        model = self.fit(X, y_low, y_high, clusters, bounds, options, L2_penalties)

        effects = self.extract_effects(model, X)

        return model, effects

    def extract_effects(self, model, X):
        """
        Extract mutation effects from a fitted regression model and calculate their MIC values.

        Args:
            model (MeIntReg): The fitted regression model, which contains fixed-effect coefficients
                and possibly a Hessian inverse matrix for uncertainty estimation.
            X (DataFrame): Binary mutation matrix with mutations as columns.

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

        effects = pd.DataFrame(
            {
                "Mutation": X.columns,
                "effect_size": fixed_effect_coefs,
            }
        )
        # Convert effect sizes to MIC values (by reversing the log transformation)
        effects["MIC"] = self.dilution_factor ** effects["effect_size"]

        if hasattr(model, "hess_inv"):
            hess_inv_dense = model.hess_inv.todense()  # Convert to a dense matrix
            # Extract the diagonal elements corresponding to the fixed effects (log(MIC) scale)
            effect_std_log = np.sqrt(np.diag(hess_inv_dense)[:p])
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

        if ecoff is None:
            ecoff, breakpoint, _, _, _ = EcoffGenerator(
                self.samples,
                self.mutations,
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
            ecoff (float, optional): Epidemiological cutoff value for classification. If None, it will be calculated. Defaults to None.
            percentile (int/float, optional): Percentile for ECOFF calculation if ecoff is None. Defaults to 99.
            p (float, optional): Significance level for classification. Defaults to 0.95.

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


if __name__ == "__main__":
    main_regression_builder(RegressionBuilder)
