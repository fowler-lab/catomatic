import os
import json
import piezo
import argparse
import numpy as np
import pandas as pd
from .defence import validate_binary_build_inputs, validate_build_piezo_inputs
from scipy.stats import norm, binomtest, fisher_exact


class BuildCatalogue:
    """
    This class builds a mutation catalogue compatible with Piezo in a standardized format.

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
                                        Defaults to False

    """

    def __init__(
        self,
        samples,
        mutations,
        FRS=None,
        seed=None,
        test=None,
        background=None,
        p=0.95,
        tails="two",
        strict_unlock=False,
        record_ids=False,
    ):
        samples = pd.read_csv(samples) if isinstance(samples, str) else samples
        mutations = pd.read_csv(mutations) if isinstance(mutations, str) else mutations

        # Run the validation function
        validate_binary_build_inputs(
            samples, mutations, seed, FRS, test, background, p, tails, record_ids
        )

        if FRS:
            # Apply fraction read support thresholds to mutations to filter out irrelevant variants
            mutations = mutations[(mutations.FRS >= FRS)]

        # Instantiate attributes
        self.catalogue = {}
        self.entry = []
        self.record_ids = record_ids
        self.temp_ids = []
        self.test = test
        self.background = background
        self.p = 1 - p
        self.strict_unlock = strict_unlock
        self.tails = tails
        self.run_iter = True

        if seed is not None:
            # If there are seeded variants, hardcode them now
            for i in seed:
                self.add_mutation(i, "S", {"seeded": "True"})

        while self.run_iter:
            # While there are susceptible solos, classify and remove them
            self.classify(samples, mutations)

        # If no more susceptible solos, classify all R and U solos in one, final sweep
        self.classify(samples, mutations)

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

    def to_piezo(
        self,
        genbank_ref,
        catalogue_name,
        version,
        drug,
        wildcards,
        outfile,
        grammar="GARC1",
        values="RUS",
        public=True,
        for_piezo=True,
        json_dumps=True,
        include_U=True,
    ):
        """
        Exports a pizeo-compatible dataframe as a csv file.

        Parameters:
            genbank_ref (str): GenBank reference identifier.
            catalogue_name (str): Name of the catalogue.
            version (str): Version of the catalogue.
            drug (str): Target drug associated with the mutations.
            wildcards (dict): Piezo wildcard (default rules) mutations with phenotypes.
            outfile: The path to the output csv file where the catalogue will be saved.
            grammar (str, optional): Grammar used in the catalogue, default "GARC1" (no other grammar currently supported).
            values (str, optional): Prediction values, default "RUS" representing each phenotype (no other values currently supported).
            public (bool, optional): private or public call
            for_piezo (bool, optional): Whether to include the missing phenotype placeholders (only piezo requires them)

        """

        piezo_df = self.build_piezo(
            genbank_ref,
            catalogue_name,
            version,
            drug,
            wildcards,
            grammar,
            values,
            public,
            for_piezo,
            json_dumps,
            include_U,
        )
        piezo_df.to_csv(outfile)

    def build_piezo(
        self,
        genbank_ref,
        catalogue_name,
        version,
        drug,
        wildcards,
        grammar="GARC1",
        values="RUS",
        public=True,
        for_piezo=True,
        json_dumps=False,
        include_U=True,
    ):
        """
        Builds a piezo-format catalogue df from the catalogue object.

        Parameters:
            genbank_ref (str): GenBank reference identifier.
            catalogue_name (str): Name of the catalogue.
            version (str): Version of the catalogue.
            drug (str): Target drug associated with the mutations.
            wildcards (dict or path): Piezo wildcard (default rules) mutations with phenotypes.
            grammar (str, optional): Grammar used in the catalogue, default "GARC1" (no other grammar currently supported).
            values (str, optional): Prediction values, default "RUS" representing each phenotype (no other values currently supported).
            public (bool, optional): private or public call
            for_piezo (bool, optional): Whether to include the missing phenotype placeholders (only piezo requires them)
            json_dumps (bool, optional): Whether to dump evidence column into json object for piezo (e.g if in notebook, unnecessary)
            include_U (bool, optional): Whether to add unclassified mutations to catalogue

        Returns:
            self: instance with piezo_catalogue set
        """

        validate_build_piezo_inputs(
            genbank_ref, catalogue_name, version, drug,
            wildcards, grammar, values, public, for_piezo, json_dumps, include_U
        )

        # if user-called
        if public:
            # add piezo wildcards to the catalogue
            if isinstance(wildcards, str):
                # if a path is supplied, read from the file
                with open(wildcards) as f:
                    wildcards = json.load(f)
            [self.add_mutation(k, v["pred"], v["evid"]) for k, v in wildcards.items()]
            # inlcude a placeholder for each phenotype if don't exist - piezo requires all R, U, S to parse
            if for_piezo:
                if not any(v["pred"] == "R" for v in self.catalogue.values()):
                    self.add_mutation("placeholder@R1R", "R", {})
                if not any(v["pred"] == "S" for v in self.catalogue.values()):
                    self.add_mutation("placeholder@S1S", "S", {})
                if (
                    not any(v["pred"] == "U" for v in self.catalogue.values())
                    or not include_U
                ):
                    self.add_mutation("placeholder@U1U", "U", {})
            data = self.catalogue
            if include_U == False:
                data = {
                    k: v
                    for k, v in data.items()
                    if (v["pred"] != "U")
                    or (k == "placeholder@U1U")
                    or ("*" in k)
                    or ("del_0.0" in k)
                }
        else:
            # if internal:
            data = wildcards

        columns = [
            "GENBANK_REFERENCE",
            "CATALOGUE_NAME",
            "CATALOGUE_VERSION",
            "CATALOGUE_GRAMMAR",
            "PREDICTION_VALUES",
            "DRUG",
            "MUTATION",
            "PREDICTION",
            "SOURCE",
            "EVIDENCE",
            "OTHER",
        ]
        # construct the catalogue dataframe in piezo-standardised format
        piezo_catalogue = (
            pd.DataFrame.from_dict(data, orient="index")
            .reset_index()
            .rename(
                columns={
                    "index": "MUTATION",
                    "pred": "PREDICTION",
                    "evid": "EVIDENCE",
                }
            )
            .assign(
                GENBANK_REFERENCE=genbank_ref,
                CATALOGUE_NAME=catalogue_name,
                CATALOGUE_VERSION=version,
                CATALOGUE_GRAMMAR=grammar,
                PREDICTION_VALUES=values,
                DRUG=drug,
                SOURCE=json.dumps({}) if json_dumps else {},
                EVIDENCE=lambda df: df["EVIDENCE"].apply(
                    json.dumps if json_dumps else lambda x: x
                ),
                OTHER=json.dumps({}) if json_dumps else {},
            )[columns]
        )

        if public:
            # Create a temporary column for the order in self.entry
            piezo_catalogue["order"] = piezo_catalogue["MUTATION"].apply(
                lambda x: self.entry.index(x)
            )

            # Sort by PREDICTION and the temporary order column
            piezo_catalogue["PREDICTION"] = pd.Categorical(
                piezo_catalogue["PREDICTION"], categories=["S", "R", "U"], ordered=True
            )
            piezo_catalogue = piezo_catalogue.sort_values(by=["PREDICTION", "order"])

            # Drop the temporary order column
            piezo_catalogue = piezo_catalogue.drop(columns=["order"])
            piezo_catalogue = piezo_catalogue[columns]

        return piezo_catalogue

    @staticmethod
    def parse_opt():
        parser = argparse.ArgumentParser(
            description="Build a catalogue and optionally export to Piezo format."
        )
        parser.add_argument(
            "--samples", required=True, type=str, help="Path to the samples file."
        )
        parser.add_argument(
            "--mutations", required=True, type=str, help="Path to the mutations file."
        )
        parser.add_argument(
            "--FRS",
            type=float,
            default=None,
            help="Optional: Fraction Read Support threshold.",
        )
        parser.add_argument(
            "--seed", nargs="+", help="Optional: List of seed mutations."
        )
        parser.add_argument(
            "--test",
            type=str,
            choices=[None, "Binomial", "Fisher"],
            default=None,
            help="Optional: Type of statistical test to run.",
        )
        parser.add_argument(
            "--background",
            type=float,
            default=None,
            help="Optional: Background mutation rate for the binomial test.",
        )
        parser.add_argument(
            "--p",
            type=float,
            default=0.95,
            help="Significance level for statistical testing.",
        )
        parser.add_argument(
            "--strict_unlock",
            action="store_true",
            help="Enforce strict unlocking for classifications.",
        )
        parser.add_argument(
            "--record_ids",
            action="store_true",
            help="Whether to record UNIQUEIDS in the catalogue"
        )
        parser.add_argument(
            "--to_json",
            action="store_true",
            help="Flag to trigger exporting the catalogue to JSON format.",
        )
        parser.add_argument(
            "--outfile",
            type=str,
            help="Path to output file for exporting the catalogue. Used with --to_json or --to_piezo.",
        )
        parser.add_argument(
            "--to_piezo",
            action="store_true",
            help="Flag to export catalogue to Piezo format.",
        )
        parser.add_argument(
            "--genbank_ref", type=str, help="GenBank reference for the catalogue."
        )
        parser.add_argument("--catalogue_name", type=str, help="Name of the catalogue.")
        parser.add_argument("--version", type=str, help="Version of the catalogue.")
        parser.add_argument(
            "--drug", type=str, help="Drug associated with the mutations."
        )
        parser.add_argument(
            "--wildcards", type=str, help="JSON file with wildcard rules."
        )
        parser.add_argument(
            "--grammar",
            type=str,
            default="GARC1",
            help="Grammar used in the catalogue.",
        )
        parser.add_argument(
            "--values",
            type=str,
            default="RUS",
            help="Values used for predictions in the catalogue.",
        )
        parser.add_argument(
            "--for_piezo",
            action="store_true",
            help="If not planning to use piezo, set to False to avoid placeholder rows being added",
        )
        return parser.parse_args()


def main():
    args = BuildCatalogue.parse_opt()
    catalogue = BuildCatalogue(
        samples=args.samples,
        mutations=args.mutations,
        FRS=args.FRS,
        seed=args.seed,
        test=args.test,
        background=args.background,
        p=args.p,
        strict_unlock=args.strict_unlock,
        record_ids=args.record_ids
    )

    if args.to_json:
        if not args.outfile:
            print("Please specify an output file with --outfile when using --to_json")
            exit(1)
        catalogue.to_json(args.outfile)

    if args.to_piezo:
        if not all(
            [
                args.genbank_ref,
                args.catalogue_name,
                args.version,
                args.drug,
                args.wildcards,
                args.outfile,
            ]
        ):
            print("Missing required arguments for exporting to Piezo format.")
            exit(1)
        catalogue.to_piezo(
            genbank_ref=args.genbank_ref,
            catalogue_name=args.catalogue_name,
            version=args.version,
            drug=args.drug,
            wildcards=args.wildcards,
            outfile=args.outfile,
            grammar=args.grammar,
            values=args.values,
            for_piezo=args.for_piezo,
        )


if __name__ == "__main__":
    main()
