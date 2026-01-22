import os
import json
import piezo
import numpy as np
import pandas as pd
from typing import Any, Optional, Tuple, List, Callable, Literal, MutableMapping, cast
from pathlib import Path
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

    samples: pd.DataFrame
    mutations: pd.DataFrame
    catalogue: dict[str, dict]
    entry: list[str]
    temp_ids: list[str]
    run_iter: bool
    seed: Optional[list] = None
    record_ids: bool
    min_count: int
    test: Optional[Literal["Binomial", "Fisher"]]
    background: Optional[float]
    p: float
    tails: Literal["one", "two"]
    strict_unlock: bool
    Contingency = List[List[int]]

    def __init__(
        self,
        samples: pd.DataFrame | str,
        mutations: pd.DataFrame | str,
        frs: Optional[float] = None,
        seed: Optional[list] = None,
    ) -> None:
        """
        Initialize the builder with sample and mutation tables.

        Args:
            samples: DataFrame or path to CSV with columns ['UNIQUEID', 'PHENOTYPE'].
            mutations: DataFrame or path to CSV with columns ['UNIQUEID', 'MUTATION'] and optional 'FRS'.
            frs: Optional FRS threshold to filter mutation rows.
            seed: Optional list of seeded mutations to pre-add.

        Returns:
            None
        """

        samples = pd.read_csv(samples) if isinstance(samples, str) else samples
        mutations = pd.read_csv(mutations) if isinstance(mutations, str) else mutations

        # Run the validation function
        validate_binary_init(samples, mutations, seed, frs)

        if frs:
            # Apply fraction read support thresholds to mutations to filter out irrelevant variants
            mutations = mutations[(mutations.FRS >= frs)]

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
        test: Optional[Literal["Binomial", "Fisher"]] = None,
        background: Optional[float] = None,
        p: float = 0.95,
        min_count: int = 0,
        tails: Literal["one", "two"] = "two",
        strict_unlock: bool = False,
        record_ids: bool = False,
    ) -> "BinaryBuilder":
        """

        Orchestrate catalogue construction and classification.

        Args:
            test: 'Binomial', 'Fisher', or None for no hypothesis testing.
            background: Background rate for binomial test (required if test == 'Binomial').
            p: Confidence parameter (default 0.95).
            min_count: Minimum samples required to consider a mutation.
            tails: 'one' or 'two' tailed test.
            strict_unlock: If True, requires statisitcal significance to classify 'S' iteratively (otherwise homogeneity suffices).
            record_ids: If True, include sample UNIQUEIDs in evidence objects.

        Returns:
            self: The built BinaryBuilder instance
        """

        validate_binary_build_inputs(test, background, p, tails, record_ids)

        self.test = test
        self.background = background
        self.strict_unlock = strict_unlock
        self.p = 1 - p
        self.tails = tails
        self.min_count = min_count
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

        self.order_catalogue()

        return self

    def classify(self, samples: pd.DataFrame, mutations: pd.DataFrame) -> None:
        """
        Orchestrate one classification iteration over exposed 'solo' mutations.

        Args:
            samples: Samples DataFrame with columns ['UNIQUEID', 'PHENOTYPE'].
            mutations: Mutations DataFrame with columns ['UNIQUEID', 'MUTATION'].

        Returns:
            None
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

        for mut in solos[(~solos.MUTATION.isna())].MUTATION.unique():

            self._process_solos(solos, mut)

        if len(self.catalogue) == classified:
            # there may be susceptible solos, but if none pass the statistical test, it can get jammed
            self.run_iter = False

    def _process_solos(self, solos: pd.DataFrame, mut: str) -> None:
        """
        Send a mutation's solos to the correct classifier.

        Args:
            solos: DataFrame of solo occurrences
            mut: the mutation identifier

        Returns:
            None
        """

        # Skip mutations with fewer than min_count samples
        if solos[solos.MUTATION == mut].shape[0] < self.min_count:
            return
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
        else:
            raise ValueError(f"Unknown test mode: {self.test}")

    def skeleton_build(self, mutation: str, x: Contingency) -> None:
        """
        Record descriptive statistics and optionally mark susceptible solos.
        Calls homogenous susceptible S.

        Args:
            mutation: Mutation identifier.
            x: [[R_count, S_count], [background_R, background_S]].

        Returns:
            None
        """

        proportion = self.calc_proportion(x)
        ci = self.calc_confidence_interval(x)

        data = {"proportion": proportion, "confidence": ci, "contingency": x}

        if self.run_iter:
            # if iteratively classifing S variants
            if proportion == 0:
                self.add_mutation(mutation, "S", data)

        else:
            # not phenotyping, just adding to catalogue
            self.add_mutation(mutation, "U", data)

    def binomial_build(self, mutation: str, x: Contingency) -> None:
        assert self.background is not None, "background must be provided for Binomial test"
        bg: float = float(self.background)

        # p-value function for binomial
        def pvalue_fn(x) -> float:
            hits: int = int(x[0][0])
            n: int = int(x[0][0] + x[0][1])
            if self.tails == "one":
                return float(binomtest(hits, n, bg, alternative="greater").pvalue)
            return float(binomtest(hits, n, bg, alternative="two-sided").pvalue)

        # susceptible_rule: when p_calc < self.p we also require proportion <= background
        def susceptible_rule(proportion: float, p_calc: float, x) -> bool:
            return bool(proportion <= bg)

        # resistant_rule: in final mode, classify R only if proportion > background
        def resistant_rule(proportion: float, p_calc: float, x) -> bool:
            return bool(proportion > bg)

        self.hypothesis_test(mutation, x, pvalue_fn, susceptible_rule, resistant_rule)


    def fishers_build(self, mutation: str, x: Contingency) -> None:
        """
        Classify mutation using Fisher's exact test and directional inference.

        Args:
            mutation: Mutation identifier.
            x: [[R_count, S_count], [background_R, background_S]].

        Returns:
            None
        """

        # p-value function for Fisher
        def pvalue_fn(x) -> Any:
            if self.tails == "one":
                _, p = fisher_exact(x, alternative="greater")
                return p
            _, p = fisher_exact(x)
            return p

        # susceptible_rule: when p_calc < self.p we require odds_ratio <= 1 to call S
        def susceptible_rule(proportion, p_calc, x) -> bool:
            odds = self.calc_odds_ratio(x)
            return odds <= 1

        # resistant_rule: in final mode, classify R only if odds_ratio > 1
        def resistant_rule(proportion, p_calc, x) -> bool:
            odds = self.calc_odds_ratio(x)
            return odds > 1

        self.hypothesis_test(mutation, x, pvalue_fn, susceptible_rule, resistant_rule)

    def hypothesis_test(
        self,
        mutation: str,
        x: Contingency,
        pvalue_fn: Callable[[Contingency], float],
        susceptible_rule: Callable[[float, float, Contingency], bool],
        resistant_rule: Callable[[float, float, Contingency], bool],
    ) -> None:
        """
        Shared decision logic for hypothesis-based classification.

        Args:
            mutation: Mutation identifier.
            x: contingency table [[R_count, S_count], [R_no_mut, S_no_mut]].
            pvalue_fn: Function that returns p-value given contingency `x`.
            susceptible_rule: Callable deciding when to call 'S' in iterative mode.
                            Signature: (proportion, p_calc, x) -> bool
            resistant_rule: Callable deciding when to call 'R' in final (non-iterative)
                            mode when p_calc < self.p. Signature: (proportion, p_calc, x) -> bool

        Returns:
            None
        """
        proportion = self.calc_proportion(x)
        ci = self.calc_confidence_interval(x)
        p_calc = pvalue_fn(x)

        data = {
            "proportion": proportion,
            "confidence": ci,
            "p_value": p_calc,
            "contingency": x,
        }

        # ITERATIVE MODE (we actively try to find susceptibles)
        if self.run_iter:
            if self.tails == "two":
                # two-tailed iterative rules
                if proportion == 0:
                    # special-case homogeneous susceptibles
                    if not self.strict_unlock:
                        self.add_mutation(mutation, "S", data)
                        return
                    # strict path falls through to p-value based rule
                    if p_calc < self.p and susceptible_rule(proportion, p_calc, x):
                        self.add_mutation(mutation, "S", data)
                        return
                else:
                    # non-zero proportion: test p-value and then apply provided rule
                    if p_calc < self.p and susceptible_rule(proportion, p_calc, x):
                        self.add_mutation(mutation, "S", data)
                        return

            else:
                # one-tailed iterative rule (classify S when there's no evidence of resistance)
                if p_calc >= self.p:
                    self.add_mutation(mutation, "S", data)
                    return

        # FINAL (NON-ITERATIVE) MODE: decide R / U
        if self.tails == "two":
            if p_calc < self.p:
                # evidence of difference — ask strategy whether it's R
                if resistant_rule(proportion, p_calc, x):
                    self.add_mutation(mutation, "R", data)
                    return
            # no difference -> unknown
            self.add_mutation(mutation, "U", data)
        else:
            # one-tailed: evidence -> R
            if p_calc < self.p:
                self.add_mutation(mutation, "R", data)

    def add_mutation(
        self, mutation: str, prediction: str, evidence: dict[str, Any]
    ) -> None:
        """
        Adds mutation to the catalogue instance, and indexes to track order.

        Args:
            mutation: Mutation identifier.
            prediction: Phenotype label, e.g., 'R', 'S', or 'U'.
            evidence: Evidence metadata for the entry.

        Returns:
            None
        """
        # add ids to catalogue if specified
        if self.record_ids and "seeded" not in evidence:
            evidence["ids"] = self.temp_ids

        self.catalogue[mutation] = {"pred": prediction, "evid": evidence}
        # record entry once mutation is added
        self.entry.append(mutation)

    def calc_confidence_interval(self, x: Contingency) -> Tuple[float, float]:
        """
        Compute a Wilson confidence interval for the resistance proportion.

        Args:
            x: [[R_count, S_count], [background_R, background_S]].

        Returns:
            (lower, upper) confidence interval tuple.
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
    def build_contingency(
        solos: pd.DataFrame, mutation: str
    ) -> Tuple[list[list[int]], list[str]]:
        """
        Build contingency counts and return IDs for a given mutation among solos.

        Args:
            solos: DataFrame of solo occurrences (one mutation per UNIQUEID).
            mutation: Mutation identifier.

        Returns:
            (contingency, ids) where contingency is [[R_count, S_count], [R_no_mut, S_no_mut]] and ids is list of UNIQUEIDs.
        """

        R_count = len(solos[(solos.PHENOTYPE == "R") & (solos.MUTATION == mutation)])
        S_count = len(solos[(solos.PHENOTYPE == "S") & (solos.MUTATION == mutation)])

        R_count_no_mut = len(solos[(solos.MUTATION.isna()) & (solos.PHENOTYPE == "R")])
        S_count_no_mut = len(solos[(solos.MUTATION.isna()) & (solos.PHENOTYPE == "S")])

        ids = solos[solos.MUTATION == mutation]["UNIQUEID"].tolist()

        return [[R_count, S_count], [R_count_no_mut, S_count_no_mut]], ids

    @staticmethod
    def calc_odds_ratio(x: Contingency) -> float:
        """
        Compute odds ratio using a 0.5 continuity correction.

        Args:
            x: [[a, b], [c, d]] representing counts.

        Returns:
            Computed odds ratio (float).
        """

        # with continuity correction
        a = x[0][0] + 0.5
        b = x[0][1] + 0.5
        c = x[1][0] + 0.5
        d = x[1][1] + 0.5

        # Calculate odds ratio
        return (a * d) / (b * c)

    @staticmethod
    def calc_proportion(x: Contingency) -> float:
        """
        Return the fraction of resistant hits from the primary cell.

        Args:
            x: [[R_count, S_count], ...].

        Returns:
            Proportion (float); returns 0.0 if denominator is zero.
        """

        return x[0][0] / (x[0][0] + x[0][1])

    def update_catalogue(
        self,
        rules: dict[str, str],
        wildcards: Optional[str] = None,
        replace: bool = False,
    ) -> "BinaryBuilder":
        """
        Updates the catalogue with the supplied expert rules, handling both individual and aggregate cases.
        If the rule is a mutation, then it is either added (if new) or replaces the existing variant. If an
        aggregate rule, then it can be either added (and piezo phenotypes will prioritise lower-level variants),
        or it can replace all variants that fall under that rule

        Args:
            rules: Mapping of rule -> phenotype (e.g., {'mut': 'R'}).
            wildcards: Path or mapping of wildcard rules (required if replace True).
            replace: If True, replace entries that match aggregate rules.

        Returns:
            The same BinaryBuilder instance (self).
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
                    with open(wildcards) as f:
                        wildcards_map = cast(
                            MutableMapping[str, dict[str, Any]],
                            json.load(f),
                        )
                elif wildcards is None:
                    wildcards_map = {}
                else:
                    wildcards_map = dict(wildcards)

                # --- now it is SAFE to mutate ---
                wildcards_map[rule] = {"pred": "R", "evid": {}}
                self.build_piezo(
                    " ", " ", " ", "temp", wildcards_map, public=False, json_dumps=True
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

    def order_catalogue(self) -> "BinaryBuilder":
        """
        Order the catalogue by insertion

        Returns:
            self: catalogue builder instance with ordered catalogue
        """

        # Return the catalogue sorted by the order in which mutations were added
        self.catalogue = {
            key: self.catalogue[key] for key in self.entry if key in self.catalogue
        }

        return self

    def to_json(self, outfile: str | Path) -> None:
        """
        Write the catalogue to a JSON file.

        Args:
            outfile: Path to output JSON file.

        Returns:
            None
        """
        with open(outfile, "w") as f:
            json.dump(self.catalogue, f, indent=4)
