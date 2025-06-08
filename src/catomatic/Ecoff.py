import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from intreg.intreg import IntReg
from .defence_module import validate_ecoff_inputs
import multiprocessing as mp
from joblib import Memory
import piezo

memory = Memory(location=".piezo_cache", verbose=0)
memory.clear(warn=False)

class EcoffGenerator:
    """
    Generate ECOFF values for wild-type samples using interval regression.
    """

    def __init__(
        self,
        samples,
        mutations=None,
        gWT_definition=None,
        dilution_factor=2,
        censored=True,
        tail_dilutions=1,
        catalogue_path=None
    ):
        """
        Initialize the ECOFF generator with sample and mutation data.

        Args:
            samples (DataFrame): DataFrame containing 'UNIQUEID' and 'MIC' columns.
            mutations (DataFrame): DataFrame containing 'UNIQUEID' and 'MUTATION' columns. Required if constraining to gWT.
            gWT_definition (str): The protocol for determining genetically wild type samples. None will not filter for WT.
            dilution_factor (int): The factor for dilution scaling (default is 2 for doubling).
            censored (bool): Flag to indicate if censored data is used.
            tail_dilutions (int): Number of dilutions to extend for interval tails if uncensored.
            catalogue_path (str): Path to catalogue file - require dif using ERJ2022 gWT definition
        """

        samples = pd.read_csv(samples) if isinstance(samples, str) else samples

        if gWT_definition is not None:
            mutations = (
                pd.read_csv(mutations) if isinstance(mutations, str) else mutations
            )

        # Run input validation
        validate_ecoff_inputs(
            samples,
            mutations,
            gWT_definition,
            dilution_factor,
            censored,
            tail_dilutions,
        )

        # Instantiate objective MIC df
        if gWT_definition is None:
            self.obj_df = samples

        elif gWT_definition == "test1":
            self.df = pd.merge(samples, mutations, how="left", on=["UNIQUEID"])
            self.df["WT"] = self.flag_test1_wt(self.df)
            self.obj_df = self.df[~self.df.WT]

        elif gWT_definition == "ERJ2022":
            samples["WT"] = self.is_erj2022_wt(mutations, samples, catalogue_path)
            self.df = samples.copy()
            self.obj_df = self.df[self.df.WT]

        # Set parameters
        self.dilution_factor = dilution_factor
        self.censored = censored
        self.tail_dilutions = tail_dilutions

    def flag_test1_wt(self, df):
        """
        Identify and flag gWT samples based on the absence of non-synonymous mutations in the resistance genes.
        """
        synonymous_ids, wt_ids = set(), set()

        # Group by 'UNIQUEID' to check mutations
        for unique_id, group in df.groupby("UNIQUEID"):
            mutations = group.MUTATION.dropna()
            if mutations.empty:  # No mutations indicate wild-type
                wt_ids.add(unique_id)
            elif all(m.split("@")[-1][0] == m.split("@")[-1][-1] for m in mutations):
                synonymous_ids.add(unique_id)  # All mutations are synonymous

        # Mark as mutant if not in wild-type or synonymous sets
        return df["UNIQUEID"].isin(synonymous_ids | wt_ids)

    def is_erj2022_wt(self, mutations, samples, catalogue_path):
        """
        Identify and flag gWT samples based on ERJ 2022 WT protocol
        """
        drugs = ["RIF", "INH", "EMB", "PZA", "AMI", "KAN", "LEV", "MXF", "ETH"]
        cat = pd.read_csv(catalogue_path)
        cat["GENE"] = cat["MUTATION"].apply(lambda x: x.split("@")[0])
        R_genes = cat[cat.PREDICTION == "R"].GENE.unique()
        drug_genes = {}
        for drug in drugs:
            drug_genes[drug] = cat[
                (cat.DRUG == drug) & (cat.GENE.isin(R_genes))
            ].GENE.unique()

        mutations = mutations[
            (mutations.GENE.isin(R_genes))
            & (mutations.UNIQUEID.isin(samples.UNIQUEID.unique()))
        ]

        antibiograms = self.parallel_antibiogram(
            mutations=mutations,
            genomes=samples,
            drug_genes=drug_genes,
            catalogue_path=catalogue_path,
            cores=mp.cpu_count(),
        )

        gWT_samples = {
            uid: all(status == "S" for status in results)
            for uid, results in antibiograms.items()
        }

        return samples["UNIQUEID"].map(gWT_samples)

    @staticmethod
    def parallel_antibiogram(mutations, genomes, drug_genes, catalogue_path, cores=4):
        mutations = mutations.set_index("UNIQUEID")
        mut_by_uid = {
            uid: df.reset_index() for uid, df in mutations.groupby("UNIQUEID")
        }

        tasks = []
        for uid in genomes.UNIQUEID.unique():
            iso_muts = mut_by_uid.get(uid, pd.DataFrame(columns=mutations.columns))
            tasks.append((uid, iso_muts, drug_genes, catalogue_path))

        ctx = mp.get_context("fork")

        with ctx.Pool(min(cores, len(tasks))) as pool:
            results = list(
                pool.imap_unordered(
                    EcoffGenerator.process_antibiogram, tasks, chunksize=10
                ),
            )

        antibiograms = {uid: calls for uid, calls in results}
        return antibiograms

    @staticmethod
    @memory.cache
    def cached_predict(mutation, drug, catalogue_path):
        catalogue = piezo.ResistanceCatalogue(catalogue_path)
        result = catalogue.predict(mutation)
        return result.get(drug, "S") if isinstance(result, dict) else result

    @staticmethod
    def process_antibiogram(args):
        uid, iso_muts, drug_genes, catalogue_path = args
        results = []

        for drug, genes in drug_genes.items():
            muts = iso_muts[iso_muts.GENE.isin(genes)]

            if muts.empty:
                preds = []
            else:
                muts = muts.copy()
                muts["PRED"] = muts["MUTATION"].apply(
                    lambda var: (
                        "S"
                        if pd.isna(var)
                        else EcoffGenerator.cached_predict(var, drug, catalogue_path)
                    )
                )
                preds = muts["PRED"].tolist()

            if drug in ["RIF", "INH", "EMB", "PZA"]:
                result = "R" if ("R" in preds or "U" in preds) else "S"
            elif drug in ["AMI", "KAN", "LEV", "MXF", "ETH"]:
                result = "R" if "R" in preds else "S"
            else:
                result = "S"

            results.append(result)

        return (uid, results)

    def define_intervals(self, df=None):
        """
        Define MIC intervals based on the dilution factor and censoring settings.

        Args:
            df (DataFrame): DataFrame containing MIC data. If public access, can optionally supply a df to override the wt.

        Returns:
            tuple: Log-transformed lower and upper bounds for MIC intervals.
        """

        if df is None:
            df = self.obj_df

        df.drop_duplicates(["UNIQUEID"], inplace=True, keep="first")

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

    def fit(self, options={}):
        """
        Fit the interval regression model for wild-type samples.

        Returns:
            OptimizeResult: The result of the optimization containing fitted parameters.
        """
        # Define and log-transform intervals
        y_low, y_high = self.define_intervals()
        # Fit the model with log-transformed data
        return IntReg(y_low, y_high).fit(method="L-BFGS-B", initial_params=None, options=options)

    def filter_wts(self, mutant=False):
        """
        Filters for wt or mutant samples. Defaults to wt (which is the assumed arg for the class)
        Allows one to switch to explicilty fitting mutants for testing and devs.

        Args:
            mutant (bool): whether to filter for mutants or wt. Default to False
        """

        if mutant:
            # filter for mutants (for testing and devs)
            self.obj_df = self.df[~self.df.WT]
        else:
            self.obj_df = self.df[self.df.WT]

    def generate(self, percentile=99, run_mutants=False, options={}):
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

        model = self.fit(options=options)
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
