import argparse
import pandas as pd


def parse_opt_ecoff_generator():
    """
    Parse command-line options for the GenerateEcoff class.

    Returns:
        argparse.Namespace: Parsed arguments from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Generate ECOFF values for wild-type samples using interval regression."
    )
    parser.add_argument(
        "--samples",
        required=True,
        type=str,
        help="Path to the samples file containing 'UNIQUEID' and 'MIC' columns.",
    )
    parser.add_argument(
        "--mutations",
        required=True,
        type=str,
        help="Path to the mutations file containing 'UNIQUEID' and 'MUTATION' columns.",
    )
    parser.add_argument(
        "--dilution_factor",
        type=int,
        default=2,
        help="The factor for dilution scaling (default: 2 for doubling).",
    )
    parser.add_argument(
        "--censored",
        action="store_true",
        help="Flag to indicate if censored data is used (default: False).",
    )
    parser.add_argument(
        "--tail_dilutions",
        type=int,
        default=1,
        help="Number of dilutions to extend for interval tails if uncensored (default: 1).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99,
        help="The desired percentile for calculating the ECOFF (default: 99).",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        help="Optional path to save the ECOFF result to a file.",
    )
    return parser.parse_args()


def main_ecoff_generator():
    """
    Main function to execute ECOFF generation from the command line.
    """
    from catomatic.Ecoff import EcoffGenerator

    args = parse_opt_ecoff_generator()

    # Instantiate the GenerateEcoff class
    generator = EcoffGenerator(
        samples=args.samples,
        mutations=args.mutations,
        dilution_factor=args.dilution_factor,
        censored=args.censored,
        tail_dilutions=args.tail_dilutions,
    )

    # Generate ECOFF
    ecoff, z_percentile, mu, sigma, model = generator.generate(
        percentile=args.percentile
    )

    # Display results
    print(f"ECOFF (Original Scale): {ecoff}")
    print(f"Percentile (Log Scale): {z_percentile}")
    print(f"Mean (mu): {mu}")
    print(f"Standard Deviation (sigma): {sigma}")

    # Optionally save results
    if args.outfile:
        with open(args.outfile, "w") as f:
            f.write(
                f"ECOFF: {ecoff}\n"
                f"Percentile (Log Scale): {z_percentile}\n"
                f"Mean (mu): {mu}\n"
                f"Standard Deviation (sigma): {sigma}\n"
                f"Model: {model}\n"
            )


def parse_opt_binary_builder():
    parser = argparse.ArgumentParser(
        description="Build a catalogue using the binary frequentist approach"
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
    parser.add_argument("--seed", nargs="+", help="Optional: List of seed mutations.")
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
        help="Whether to record UNIQUEIDS in the catalogue.",
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
    return parser


def main_binary_builder():
    from catomatic.BinaryCatalogue import BinaryBuilder

    # Create main parser
    full_parser = argparse.ArgumentParser(
        description="Binary Builder CLI",
        parents=[parse_opt_binary_builder(), parse_opt_piezo_export()],  # Merge parsers
        add_help=True  # Only enable help here
    )

    args = full_parser.parse_args()  # Ensure all args are parsed together

    # Instantiate BinaryBuilder
    builder = BinaryBuilder(
        samples=args.samples,
        mutations=args.mutations,
        FRS=args.FRS,
        seed=args.seed,
    )

    builder.build(
        test=args.test,
        background=args.background,
        p=args.p,
        strict_unlock=args.strict_unlock,
        record_ids=args.record_ids,
    )

    # Handle JSON export
    if args.to_json:
        main_json_exporter(builder, args)

    # Handle Piezo export
    if args.to_piezo:
        main_piezo_exporter(builder, args)




def parse_opt_regression_builder():
    """
    Parse command-line options for the RegressionBuilder class.

    Returns:
        argparse.Namespace: Parsed arguments from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Build a regression-based mutation catalogue."
    )
    parser.add_argument(
        "--samples", required=True, type=str, help="Path to the samples file (CSV)."
    )
    parser.add_argument(
        "--mutations", required=True, type=str, help="Path to the mutations file (CSV)."
    )
    parser.add_argument(
        "--genes",
        type=list,
        default=[],
        help="A list of RAV genes. A list must be supplied if non-RAV genes are in the mutations table (ie if clustering snp distances)",
    )
    parser.add_argument(
        "--dilution_factor", type=int, default=2, help="Dilution factor (default: 2)."
    )
    parser.add_argument(
        "--censored",
        action="store_true",
        help="Indicates if the data is censored (default: False).",
    )
    parser.add_argument(
        "--tail_dilutions",
        type=int,
        default=1,
        help="Tail dilutions for uncensored data (default: 1).",
    )
    parser.add_argument(
        "--FRS",
        type=float,
        default=None,
        help="Fraction Read Support threshold (default: None).",
    )
    parser.add_argument(
        "--ecoff",
        type=float,
        default=None,
        help="Epidemiological cutoff value for classification. If None, it will be calculated",
    )
    parser.add_argument(
        "--b_bounds",
        nargs=2,
        type=float,
        default=(None, None),
        help="Bounds for beta coefficients.",
    )
    parser.add_argument(
        "--u_bounds",
        nargs=2,
        type=float,
        default=(None, None),
        help="Bounds for random effects.",
    )
    parser.add_argument(
        "--s_bounds",
        nargs=2,
        type=float,
        default=(None, None),
        help="Bounds for sigma.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99,
        help="Percentile for ECOFF calculation (default: 99).",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.95,
        help="Significance level for statistical testing (default: 0.95).",
    )
    parser.add_argument(
        "--fixed_effects",
        type=list,
        default=None,
        help="List of fixed effect column names (default: None).",
    )
    parser.add_argument(
        "--random_effects",
        action="store_true",
        help="Whether to perform SNP clustering and include as a random effect (default True)",
    )
    parser.add_argument(
        "--cluster_distance",
        type=float,
        default=1,
        help="Clustering distance threshold (default: 1).",
    )
    parser.add_argument(
        "--outfile", type=str, required=False, help="Path to save the output JSON file."
    )
    parser.add_argument(
        "--options",
        type=dict,
        default=None,
        help="Scipy minimise's ptimization options for the regression fitting.",
    )
    parser.add_argument(
        "--L2_penalties",
        type=dict,
        default=None,
        help="Regularization penalties for fixed and random effects",
    )
    parser.add_argument(
        "--to_json",
        action="store_true",
        help="Flag to trigger exporting the catalogue to JSON format.",
    )
    return parser


def main_regression_builder():
    """
    Main function to build the regression-based mutation catalogue and handle CLI options.
    """
    from catomatic.RegressionCatalogue import RegressionBuilder
    # Parse CLI arguments for regression builder
    regression_parser = parse_opt_regression_builder()
    args, unknown = regression_parser.parse_known_args()

    # Parse piezo arguments separately
    piezo_parser = parse_opt_piezo_export()
    piezo_args = piezo_parser.parse_args(unknown)

    # Instantiate RegressionBuilder and build the catalogue
    builder = RegressionBuilder(
        samples=args.samples,
        mutations=args.mutations,
        genes=args.genes,
        dilution_factor=args.dilution_factor,
        censored=args.censored,
        tail_dilutions=args.tail_dilutions,
        FRS=args.FRS,
    )

    builder.build(
        b_bounds=args.b_bounds,
        u_bounds=args.u_bounds,
        s_bounds=args.s_bounds,
        options=args.options,
        L2_penalties=args.L2_penalties,
        ecoff=args.ecoff,
        percentile=args.percentile,
        p=args.p,
        fixed_effects=args.fixed_effects,
        random_effects=args.random_effects,
        cluster_distance=args.cluster_distance,
    )

    # Handle JSON export
    if args.to_json:
        main_json_exporter(builder, args)

    # Handle Piezo export if enabled
    if piezo_args.to_piezo:
        main_piezo_exporter(builder, piezo_args)



def main_json_exporter(builder, args):
    if not args.outfile:
        print("Please specify an output file with --outfile when using --to_json")
        exit(1)
    builder.to_json(args.outfile)
    print(f"Catalogue exported to {args.outfile}")


def main_piezo_exporter(builder, piezo_args):
    if not all(
        [
            piezo_args.genbank_ref,
            piezo_args.catalogue_name,
            piezo_args.version,
            piezo_args.drug,
            piezo_args.wildcards,
            piezo_args.outfile,
        ]
    ):
        print("Missing required arguments for Piezo export.")
        exit(1)
    builder.to_piezo(
        genbank_ref=piezo_args.genbank_ref,
        catalogue_name=piezo_args.catalogue_name,
        version=piezo_args.version,
        drug=piezo_args.drug,
        wildcards=piezo_args.wildcards,
        outfile=piezo_args.outfile,
        grammar=piezo_args.grammar,
        values=piezo_args.values,
        for_piezo=piezo_args.for_piezo,
    )
    print("Catalogue exported to Piezo format.")


def parse_opt_piezo_export():
    parser = argparse.ArgumentParser(
        description="Export the catalogue in piezo standard format",
        add_help=False  # Disable auto `-h/--help` to prevent conflicts
    )
    parser.add_argument("--to_piezo", action="store_true", help="Flag to export catalogue to Piezo format.")
    parser.add_argument("--genbank_ref", type=str, help="GenBank reference for the catalogue.")
    parser.add_argument("--catalogue_name", type=str, help="Name of the catalogue.")
    parser.add_argument("--version", type=str, help="Version of the catalogue.")
    parser.add_argument("--drug", type=str, help="Drug associated with the mutations.")
    parser.add_argument("--wildcards", type=str, help="JSON file with wildcard rules.")
    parser.add_argument("--grammar", type=str, default="GARC1", help="Grammar used in the catalogue.")
    parser.add_argument("--values", type=str, default="RUS", help="Values used for predictions in the catalogue.")
    parser.add_argument("--for_piezo", action="store_true",
                        help="If not planning to use piezo, set to False to avoid placeholder rows being added")
    
    return parser
