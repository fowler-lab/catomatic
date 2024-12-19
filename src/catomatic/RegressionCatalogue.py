import numpy as np
import pandas as pd
from intreg.meintreg import MeIntReg
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering


class BuildRegressionCatalogue:
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
        self.samples = samples
        self.df = pd.merge(samples, mutations, how="left", on=["UNIQUEID"])

        # Set parameters
        self.dilution_factor = dilution_factor
        self.censored = censored
        self.tail_dilutions = tail_dilutions
        self.cluster_distance = cluster_distance

    def build_X(self, df):
        """
        Build the binary mutation matrix X.

        Args:
            df (DataFrame): Merged DataFrame of sample and mutation data.

        Returns:
            DataFrame: Binary mutation matrix (1 for presence, 0 for absence).
        """
        # IDs to reindex after creating the matrix
        ids = df.UNIQUEID.unique()

        X = pd.pivot_table(
            df,
            index="UNIQUEID",
            columns="MUTATION",
            aggfunc="size",  # counts occurrences
            fill_value=0,  # absence of the mutation
        )

        # Convert counts to binary presence/absence (1/0)
        X = X.map(lambda x: 1 if x > 0 else 0)
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

        # Calculate Hamming distances
        dist_matrix = pairwise_distances(X, metric="hamming")

        # Agglomerative clustering
        agg_cluster = AgglomerativeClustering(
            metric="precomputed",
            linkage="complete",
            distance_threshold=self.cluster_distance
            / len(X.columns),  # Hamming threshold conversion
            n_clusters=None,
        )
        # Cluster IDs for each sample
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

        # Calculate tail dilution factor if not censored
        if not self.censored:
            tail_dilution_factor = self.dilution_factor**self.tail_dilutions

        # Process each MIC value and define intervals
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
        # Transform intervals to log space
        y_low_log = np.log(y_low, where=(y_low > 0)) / log_base
        y_high_log = np.log(y_high, where=(y_high > 0)) / log_base

        return y_low_log, y_high_log

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
        beta_init = np.random.normal(loc=0, scale=0.5, size=p)
        u_init = np.zeros(len(np.unique(clusters)))
        # sigma = 1  # Fixed initial sigma
        sigma = np.random.uniform(0.5, 2.0)
        # sigma = np.random.uniform(0.5, 2.0)

        return beta_init, u_init, sigma

    def fit(self, X, y_low, y_high, clusters, bounds=None, options=None):
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

        if options is not None:
            return MeIntReg(y_low, y_high, X, clusters).fit(
                method="L-BFGS-B",
                initial_params=initial_params,
                bounds=bounds,
                options=options,
            )
        else:
            return self.iter_tolerances(
                X, y_low, y_high, clusters, initial_params, bounds
            )

    def iter_tolerances(self, X, y_low, y_high, clusters, initial_params, bounds):
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

        #may need to reduce maxfun search for speed up.
        #maxfun (number function evaluations) is generally too low 
        # (default 15000) to fit, so can get a success either by 
        # increasing or by loosening tolerances. Below tries to find a balance.

        gtols = [1e-5, 1e-4, 1e-3]
        ftols = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

        for gtol in gtols:
            for ftol in ftols:
                r = MeIntReg(y_low, y_high, X, clusters).fit(
                    method="L-BFGS-B",
                    initial_params=initial_params,
                    bounds=bounds,
                    options={
                        "maxiter": 10000,
                        "maxfun": 100000,
                        "ftol": ftol,
                        "gtol": gtol,
                    },
                )
                if r.success:
                    return r

    def predict_effects(
        self,
        b_bounds=(None, None),
        u_bounds=(None, None),
        s_bounds=(None, None),
        options=None,
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
            options (dict or None): options for scipy minimise

        Returns:
            tuple: Fitted regression model and mutation matrix X.
        """
        y_low, y_high = self.define_intervals(self.samples)
        X = self.build_X(self.df)
        clusters = self.cluster_coords(X)

        b_bounds = [b_bounds] * X.shape[1]
        u_bounds = [u_bounds] * len(np.unique(clusters))
        bounds = b_bounds + u_bounds + [s_bounds]

        model = self.fit(X, y_low, y_high, clusters, bounds, options)

        return model, X
