import numpy as np
import pandas as pd
from scipy.stats import norm
from intreg.intreg import IntReg
from .defence_module import validate_ecoff_inputs
from .cli_module import main_ecoff_generator


class EcoffGenerator:
    """
    Generate ECOFF values for wild-type samples using interval regression.
    """

    def __init__(
        self, samples, mutations, dilution_factor=2, censored=True, tail_dilutions=1
    ):
        """
        Initialize the ECOFF generator with sample and mutation data.

        Args:
            samples (DataFrame): DataFrame containing 'UNIQUEID' and 'MIC' columns.
            mutations (DataFrame): DataFrame containing 'UNIQUEID' and 'MUTATION' columns.
            dilution_factor (int): The factor for dilution scaling (default is 2 for doubling).
            censored (bool): Flag to indicate if censored data is used.
            tail_dilutions (int): Number of dilutions to extend for interval tails if uncensored.
        """

        samples = pd.read_csv(samples) if isinstance(samples, str) else samples
        mutations = pd.read_csv(mutations) if isinstance(mutations, str) else mutations

        # Run input validation
        validate_ecoff_inputs(
            samples, mutations, dilution_factor, censored, tail_dilutions
        )

        # Merge data and flag mutants
        self.df = pd.merge(samples, mutations, how="left", on=["UNIQUEID"])
        self.flag_mutants()
        self.filter_mutants()

        # Set parameters
        self.dilution_factor = dilution_factor
        self.censored = censored
        self.tail_dilutions = tail_dilutions


    def flag_mutants(self):
        """
        Identify and flag mutant samples based on the presence of mutations.
        """
        synonymous_ids, wt_ids = set(), set()

        # Group by 'UNIQUEID' to check mutations
        for unique_id, group in self.df.groupby("UNIQUEID"):
            mutations = group.MUTATION.dropna()
            if mutations.empty:  # No mutations indicate wild-type
                wt_ids.add(unique_id)
            elif all(m.split("@")[-1][0] == m.split("@")[-1][-1] for m in mutations):
                synonymous_ids.add(unique_id)  # All mutations are synonymous

        # Mark as mutant if not in wild-type or synonymous sets
        self.df["MUTANT"] = ~self.df["UNIQUEID"].isin(synonymous_ids | wt_ids)

    def define_intervals(self, df=None):
        """
        Define MIC intervals based on the dilution factor and censoring settings.

        Args:
            df (DataFrame): DataFrame containing MIC data. If public access, can optionally supply a df to override the wt.

        Returns:
            tuple: Log-transformed lower and upper bounds for MIC intervals.
        """

        if df is None:
            df = self.wt_df

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

    def fit(self):
        """
        Fit the interval regression model for wild-type samples.

        Returns:
            OptimizeResult: The result of the optimization containing fitted parameters.
        """
        # Define and log-transform intervals
        y_low, y_high = self.define_intervals()
        # Fit the model with log-transformed data
        return IntReg(y_low, y_high).fit(method="L-BFGS-B", initial_params=None)
    
    def filter_mutants(self, mutant=False):
        """
        Filters for wt or mutant samples. Defaults to wt (which is the assumed arg for the class)
        Allows one to switch to explicilty fitting mutants for testing and devs.

        Args:
            mutant (bool): whether to filter for mutants or wt. Default to False
        """

        if mutant:
            #filter for mutants (for testing and devs)
            self.wt_df = self.df[self.df.MUTANT]
        else:
            self.wt_df = self.df[~self.df.MUTANT]
        
    def generate(self, percentile=99, run_mutants=False):
        """
        Calculate the ECOFF value based on the fitted model and a specified percentile.

        Args:
            percentile (float): The desired percentile (e.g., 99 for 99th percentile).

        Returns:
            tuple: ECOFF in the original scale, the specified percentile in the log-transformed scale,
                   mean (mu), standard deviation (sigma), and the model result.
        """

        assert (
            percentile > 0 and percentile < 100
        ), "percentile must be a float or integer between 0 and 100"

        model = self.fit()
        # Extract model parameters
        mu, log_sigma = model.x
        sigma = np.exp(log_sigma)
        # Calulcate z-score for the given percentile
        z_score = norm.ppf(percentile / 100)
        # Calculate the percentile in log scale
        z_percentile = mu + z_score * sigma
        # Convert the percentile back to the original MIC scale
        ecoff = self.dilution_factor**z_percentile

        return ecoff, z_percentile, mu, sigma, model


if __name__ == "__main__":
    main_ecoff_generator(EcoffGenerator)
