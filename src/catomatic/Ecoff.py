import numpy as np
import pandas as pd
from intreg.intreg import IntReg
from .defence import validate_ecoff_inputs



class GenerateEcoff:
    """
    Generate ECOFF values for wild-type samples using interval regression.
    """

    def __init__(self, samples, mutations, dilution_factor=2, censored=True, tail_dilutions=None):
        """
        Initialize the ECOFF generator with sample and mutation data.

        Args:
            samples (DataFrame): DataFrame containing 'UNIQUEID' and 'MIC' columns.
            mutations (DataFrame): DataFrame containing 'UNIQUEID' and 'MUTATION' columns.
            dilution_factor (int): The factor for dilution scaling (default is 2 for doubling).
            censored (bool): Flag to indicate if censored data is used.
            tail_dilutions (int): Number of dilutions to extend for interval tails if uncensored.
        """
        # Run input validation
        validate_ecoff_inputs(samples, mutations, dilution_factor, censored, tail_dilutions)

        # Merge data and flag mutants
        self.df = pd.merge(samples, mutations, how="left", on=["UNIQUEID"])
        self.flag_mutants()

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
            tail_dilution_factor = self.dilution_factor ** self.tail_dilutions

        # Process each MIC value and define intervals
        for i, mic in enumerate(df.MIC):
            if mic.startswith("<="):  # Left-censored
                lower_bound = float(mic[2:])
                y_low[i] = 1e-6 if self.censored else lower_bound / tail_dilution_factor
                y_high[i] = lower_bound
            elif mic.startswith(">"):  # Right-censored
                upper_bound = float(mic[1:])
                y_low[i] = upper_bound
                y_high[i] = np.inf if self.censored else upper_bound * tail_dilution_factor
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
        # Filter out mutant samples
        self.wt_df = self.df[self.df.MUTANT == False]
        # Define and log-transform intervals
        y_low, y_high = self.define_intervals(self.wt_df)
        # Fit the model with log-transformed data
        return IntReg(y_low, y_high).fit(method="L-BFGS-B", initial_params=None)

    def generate(self):
        """
        Calculate the ECOFF value based on the fitted model.

        Returns:
            tuple: ECOFF in the original scale, the 99th percentile in the log-transformed scale, 
                   mean (mu), standard deviation (sigma), and the model result.
        """
        
        model = self.fit()
        # Extract model parameters
        mu, log_sigma = model.x
        sigma = np.exp(log_sigma)
        # Calculate the 99th percentile (z_99) in log scale
        z_99 = mu + 2.3263 * sigma
        # Convert z_99 back to original MIC scale
        ecoff = self.dilution_factor ** z_99

        return ecoff, z_99, mu, sigma, model
