import os
import json
import piezo
import argparse
import numpy as np
import pandas as pd
from .PiezoTools import PiezoExporter
from .defence_module import validate_binary_init, validate_binary_build_inputs
from scipy.stats import norm, binomtest, fisher_exact


class BinaryBuilder(PiezoExporter):
    """
    Builds a mutation catalogue compatible with Piezo in a standardized format.

    Binary labels underpin a frequentist statistical approach.

    Instantiation constructs the catalogue object.

    Parameters:
        samples (pd.DataFrame): A DataFrame containing sample identifiers along with a binary
                                'R' vs 'S' phenotype for each sample.
                                Required columns: ['UNIQUEID', 'PHENOTYPE']

        mutations (pd.DataFrame): A DataFrame containing mutations in relevant genes for each sample.
                                  Required columns: ['UNIQUEID', 'MUTATION']
                                  Optional columns: ['FRS']

        FRS (float, optional): The Fraction Read Support threshold used to construct the catalogues.
                               Lower FRS values allow for greater genotype heterogeneity.

        seed (list) optional): A list of predefined GARC neutral mutations with associated phenotypes
                               that are hardcoded prior to running the builder. Defaults to None.

    """

    def __init__(
        self,
        samples,
        mutations,
        FRS=None,
        seed=None,
    ):
        samples = pd.read_csv(samples) if isinstance(samples, str) else samples
        mutations = pd.read_csv(mutations) if isinstance(mutations, str) else mutations

        # Run the validation function
        validate_binary_init(samples, mutations, seed, FRS)

        if FRS:
            # Apply fraction read support thresholds to mutations to filter out irrelevant variants
            mutations = mutations[(mutations.FRS >= FRS)]

        self.samples = samples
        self.mutations = mutations

        # Instantiate attributes
        self.catalogue = {}
        self.entry = []
        self.temp_ids = []
        self.run_iter = True
        self.seed = seed

    def build(
        self,
        test=None,
        background=None,
        p=0.95,
        tails="two",
        strict_unlock=False,
        record_ids=False,
    ):
        """
        Args:
        test (str, optional): Type of statistical test to run for phenotyping. None (doesn't phenotype)
                                vs binomial (against a defined background) vs Fisher (against contingency
                                background). Defaults to none.

        background (float, optional): Background rate between 0-1 for binomial test phenotyping. Deafults to None.

        p (float, optional): Significance level at which to reject the null hypothesis during statistical testing.
                             Defaults to 0.95.
        tails (str, optional): Whether to run a 1-tailed or 2-tailed test. Defaults to 'two'.
        strict_unlock (bool, optional): If strict_unlock is true,  statistical significance in the direction of
                                        susceptiblity will be required for S classifications. If false, homogenous
                                        susceptiblity is sufficient for S classifcations. Defaults to False
        record_ids (bool, optional): If true, will track identifiers to which the mutations belong and were extracted
                                        from - helpful for detailed interrogation, but gives long evidence objects.
                                        Defaults to False"""
        
        validate_binary_build_inputs(test, background, p, tails, record_ids)

        self.test = test
        self.background = background
        self.strict_unlock = strict_unlock
        self.p = 1 - p
        self.tails = tails
        self.record_ids = record_ids
                
        if self.seed is not None:
            # If there are seeded variants, hardcode them now
            for i in self.seed:
                self.add_mutation(i, "S", {"seeded": "True"})

        while self.run_iter:
            # While there are susceptible solos, classify and remove them
            self.classify(self.samples, self.mutations)

        # If no more susceptible solos, classify all R and U solos in one, final sweep
        self.classify(self.samples, self.mutations)

        return self

    def classify(self, samples, mutations):
        """
        Classifies susceptible mutations by extracting samples with only 1 mutation, and iterates through
        the pooled mutations to determine whether there is statistical evidence for susceptibility, for each
        unique mutation type.

        Parameters:
            samples (pd.DataFrame): A DataFrame containing sample identifiers along with a binary
                                    'R' vs 'S' phenotype for each sample.
                                    Required columns: ['UNIQUEID', 'PHENOTYPE']

            mutations (pd.DataFrame): A DataFrame containing mutations in relevant genes for each sample.
                                    Required columns: ['UNIQUEID', 'MUTATION']
        """

        # remove mutations predicted as susceptible from df (to potentially proffer additional, effective solos)
        mutations = mutations[
            ~mutations.MUTATION.isin(
                mut for mut, _ in self.catalogue.items() if _["pred"] == "S"
            )
        ]
        # left join mutations to phenotypes
        joined = pd.merge(samples, mutations, on=["UNIQUEID"], how="left")
        # extract samples with only 1 mutation
        solos = joined.groupby("UNIQUEID").filter(lambda x: len(x) == 1)

        # no solos or susceptible solos, so method is jammed - end here and move to classifying resistant variants.
        if len(solos) == 0 or all(solos.PHENOTYPE == "R"):
            self.run_iter = False

        classified = len(self.catalogue)

        # for each non-synonymous mutation type
        for mut in solos[(~solos.MUTATION.isna())].MUTATION.unique():
            # build a contingency table
            x, ids = self.build_contingency(solos, mut)
            # temporarily store mutation groups:
            self.temp_ids = ids
            # classify susceptible variants according to specified test mode
            if self.test is None:
                self.skeleton_build(mut, x)
            elif self.test == "Binomial":
                self.binomial_build(mut, x)
            elif self.test == "Fisher":
                self.fishers_build(mut, x)

        if len(self.catalogue) == classified:
            # there may be susceptible solos, but if none pass the test, it can get jammed
            self.run_iter = False

    def skeleton_build(self, mutation, x):
        """
        Calculates proportion of resistance with confidence intervals. Does not test nor
        phenotype. Assumes suscepitble solos display homogenous susceptibility.

        Parameters:
            mutation (str): mutation identifier
            x table (list): [[R count, S count],[background R, background S]]
        """

        proportion = self.calc_proportion(x)
        ci = self.calc_confidenceInterval(x)

        data = {"proportion": proportion, "confidence": ci, "contingency": x}

        if self.run_iter:
            # if iteratively classifing S variants
            if proportion == 0:
                self.add_mutation(mutation, "S", data)

        else:
            # not phenotyping, just adding to catalogue
            self.add_mutation(mutation, "U", data)

    def binomial_build(self, mutation, x):
        """
        Calculates proportion of resistance, confidence intervals, and phenotypes
        relative to a defined, assumed background rate using a binomial test.6

        Parameters:
            mutation (str): mutation identifier
            x (list): contingency table: [[R count, S count],[background R, background S]]
        """

        proportion = self.calc_proportion(x)
        ci = self.calc_confidenceInterval(x)

        # going to actively classify S - if above specified background (e.g 90%) on iteratrion
        # this is quite strict - if no difference to background, then logically should be S,
        # but we are allowing in U classifications to find those mutations on the edge or with
        # large confidence intervals.
        hits = x[0][0]
        n = x[0][0] + x[0][1]

        if self.tails == "one":
            p_calc = binomtest(hits, n, self.background, alternative="greater").pvalue
        else:
            p_calc = binomtest(hits, n, self.background, alternative="two-sided").pvalue

        data = {
            "proportion": proportion,
            "confidence": ci,
            "p_value": p_calc,
            "contingency": x,
        }

        if self.run_iter:
            # Check for iterative classification of S variants
            if self.tails == "two":
                # if two-tailed
                if proportion == 0:
                    if not self.strict_unlock:
                        # Classify S when  no evidence of resistance and homogeneous S classifications are allowed
                        self.add_mutation(mutation, "S", data)
                    elif p_calc < self.p:
                        # Classify as susceptible if statistically S (stricter)
                        if proportion <= self.background:
                            self.add_mutation(mutation, "S", data)
                elif p_calc < self.p:
                    # Classify as susceptible based on active evaluation and background proportion
                    if proportion <= self.background:
                        self.add_mutation(mutation, "S", data)
            else:
                # if one-tailed
                if p_calc >= self.p:
                    # Classify susceptible if no evidence of resistance
                    self.add_mutation(mutation, "S", data)
        else:
            if self.tails == "two":
                # if two-tailed
                if p_calc < self.p:
                    # if R, classify resistant
                    if proportion > self.background:
                        self.add_mutation(mutation, "R", data)
                else:
                    # if no difference, classify U
                    self.add_mutation(mutation, "U", data)
            else:
                # if one-tailed
                if p_calc < self.p:
                    # Classify resistance if evidence of resistance
                    self.add_mutation(mutation, "R", data)

    def fishers_build(self, mutation, x):
        """
        Determines if theres a statistically significant difference between resistant
        or susceptible hits and the calculated background rate for that mutation at that iteration,
        in the direction determined by an odds ratio. Classifies S as statistically different from background,
        or homogenous susceptibility (becauase [0, 1] p-value > 0.05)

        Parameters:
            mutation (str): mutation identifier
            x (list): contingency table [[R count, S count],[background R, background S]]
        """

        proportion = self.calc_proportion(x)
        ci = self.calc_confidenceInterval(x)

        if self.tails == "one":
            _, p_calc = fisher_exact(x, alternative="greater")
        else:
            _, p_calc = fisher_exact(x)

        data = {
            "proportion": proportion,
            "confidence": ci,
            "p_value": p_calc,
            "contingency": x,
        }

        if self.run_iter:
            # if iteratively classifing S variants
            if self.tails == "two":
                # if two-tailed
                if proportion == 0:
                    if not self.strict_unlock:
                        # Classify S when  no evidence of resistance and homogeneous S classifications are allowed
                        self.add_mutation(mutation, "S", data)
                    elif p_calc < self.p:
                        # if difference and statisitcal significance required for S classiication
                        odds = self.calc_oddsRatio(x)
                        # if S, call susceptible
                        if odds <= 1:
                            self.add_mutation(mutation, "S", data)
                elif p_calc < self.p:
                    # if different from background, calculate OR to determine direction
                    odds = self.calc_oddsRatio(x)
                    # if S, call susceptible
                    if odds <= 1:
                        self.add_mutation(mutation, "S", data)
            else:
                # if one-tailed
                if p_calc >= self.p:
                    # Classify susceptible if no evidence of resistance
                    self.add_mutation(mutation, "S", data)

        else:
            if self.tails == "two":
                # if two-sided
                if p_calc < self.p:
                    # calculate OR to determine direction
                    odds = self.calc_oddsRatio(x)
                    # if R, call resistant
                    if odds > 1:
                        self.add_mutation(mutation, "R", data)
                # if no difference, call U
                else:
                    self.add_mutation(mutation, "U", data)
            else:
                # if one-sided
                if p_calc < self.p:
                    # if there is evidence of resistance
                    self.add_mutation(mutation, "R", data)

    def add_mutation(self, mutation, prediction, evidence):
        """
        Adds mutation to cataloue object, and indexes to track order.

        Parameters:
            mutation (str): mutaiton to be added
            prediction (str): phenotype of mutation
            evidence (any): additional metadata to be added
        """
        # add ids to catalogue if specified
        if self.record_ids and "seeded" not in evidence:
            evidence["ids"] = self.temp_ids

        self.catalogue[mutation] = {"pred": prediction, "evid": evidence}
        # record entry once mutation is added
        self.entry.append(mutation)

    def calc_confidenceInterval(self, x):
        """
        Calculates Wilson confidence intervals from the proportion..

        Parameters:
            x (list): contingency table [[R count, S count],[background R, background S]]

        Returns:
        lower, upper (tuple): upper and lower bounds of confidence interval
        """

        z = norm.ppf(1 - self.p / 2)
        proportion = self.calc_proportion(x)
        n = x[0][0] + x[0][1]
        denom = 1 + (z**2 / n)
        centre_adjusted_prob = (proportion) + (z**2 / (2 * n))
        adjusted_sd = z * np.sqrt(
            ((proportion) * (1 - proportion) / n) + (z**2 / (4 * n**2))
        )

        lower = (centre_adjusted_prob - adjusted_sd) / denom
        upper = (centre_adjusted_prob + adjusted_sd) / denom

        return (lower, upper)

    @staticmethod
    def build_contingency(solos, mut):
        """
        Constructs a contingency table for a specific mutation within a df of solo occurrences.

        Parameters:
            solos (pd.DataFrame): df containing solo mutations
                                Required columns: ['MUTATION', 'PHENOTYPE']
            mut (str): The specific mutation

        Returns:
                [[R count, S count],[background R, background S]]
        """

        R_count = len(solos[(solos.PHENOTYPE == "R") & (solos.MUTATION == mut)])
        S_count = len(solos[(solos.PHENOTYPE == "S") & (solos.MUTATION == mut)])

        R_count_no_mut = len(solos[(solos.MUTATION.isna()) & (solos.PHENOTYPE == "R")])
        S_count_no_mut = len(solos[(solos.MUTATION.isna()) & (solos.PHENOTYPE == "S")])

        ids = solos[solos.MUTATION == mut]["UNIQUEID"].tolist()

        return [[R_count, S_count], [R_count_no_mut, S_count_no_mut]], ids

    @staticmethod
    def calc_oddsRatio(x):
        """
        Calculates odds ratio

        Parameters:
            x (list): contingency table [[R count, S count],[background R, background S]]

        Returns:
            Odds ratio.
        """
        # with continuity correction
        a = x[0][0] + 0.5
        b = x[0][1] + 0.5
        c = x[1][0] + 0.5
        d = x[1][1] + 0.5

        # Calculate odds ratio
        return (a * d) / (b * c)

    @staticmethod
    def calc_proportion(x):
        """
        Calculates proportion of hits

        Parameters:
            x (list): contingency table [[R count, S count],[background R, background S]]

        Returns:
            Fraction of hits.
        """

        return x[0][0] / (x[0][0] + x[0][1])

    def update(self, rules, wildcards=None, replace=False):
        """
        Updates the catalogue with the supplied expert fules, handling both individual and aggregate cases.
        If the rule is a mutation, then it is either added (if new) or replaces the existing variant. If an
        aggregate rule, then it can be either added (and piezo phenotypes will prioritise lower-level variants),
        or it can replace all variants that fall under that rule

        Parameters:
            rules (dict): A dictionary mapping rules to phenotypes. {mut:pred}.
            replace (bool, optional): If True, allows replacement of existing entries. Defaults to False.

        Returns:
            self: Returns the instance with updated catalogue.
        """

        if not os.path.exists("./temp"):
            os.makedirs("./temp")

        for rule, phenotype in rules.items():
            # if not an aggregate rule
            if "*" not in rule and rule in self.entry:
                # have to replace if already exists
                self.catalogue.pop(rule, None)
                self.entry.remove(rule)
            # if an aggregate rule, and replacement has been specified
            elif replace:
                assert (
                    wildcards is not None
                ), "wildcards must be supplied if replace is used"
                # write rule in piezo format to temp (need piezo to find vars)
                if isinstance(wildcards, str):
                    # if a path is supplied, read from the file
                    with open(wildcards) as f:
                        wildcards = json.load(f)
                wildcards[rule] = {"pred": "R", "evid": {}}
                self.build_piezo(
                    "", "", "", "temp", wildcards, public=False, json_dumps=True
                ).to_csv("./temp/rule.csv", index=False)
                # read rule back in with piezo
                piezo_rule = piezo.ResistanceCatalogue("./temp/rule.csv")
                # find variants to be replaced
                target_vars = {
                    k: v["evid"]
                    for k, v in self.catalogue.items()
                    if (("default_rule" not in v["evid"]) and (len(v["evid"]) != 0))
                    and (
                        (predict := piezo_rule.predict(k)) == "R"
                        or (isinstance(predict, dict) and predict.get("temp") == "R")
                    )
                }
                # remove those to be replaced
                for k in target_vars.keys():
                    if k in self.entry:
                        self.catalogue.pop(k, None)
                        self.entry.remove(k)
                # clean up
                os.remove("./temp/rule.csv")

            # add rule to catalogue
            self.add_mutation(rule, phenotype, {})

        return self

    def return_catalogue(self, ordered=False):
        """
        Public method that returns the catalogue dictionary, sorted either by order of addition.

        Returns:
            dict: The catalogue data stored in the instance.
        """

        # Return the catalogue sorted by the order in which mutations were added
        return {key: self.catalogue[key] for key in self.entry if key in self.catalogue}

    def to_json(self, outfile):
        """
        Exports the catalogue to a JSON file.

        Parameters:
            outfile (str): The path to the output JSON file where the catalogue will be saved.
        """
        with open(outfile, "w") as f:
            json.dump(self.catalogue, f, indent=4)

