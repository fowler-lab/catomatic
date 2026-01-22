import argparse
from catomatic.cli import (
    parse_binary_builder,
    main_binary_builder,
    parse_regression_builder,
    main_regression_builder,
)


def main():
    """
    Main function to parse command-line arguments and execute the appropriate module.
    """
    parser = argparse.ArgumentParser(
        description="Catomatic CLI - Run different catalogue builders and ECOFF generators."
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Binary Catalogue Builder
    binary_parser = subparsers.add_parser("binary", help="Build a catalogue using the binary frequentist approach.")
    for action in parse_binary_builder()._actions:
        if action.dest != "help":
            binary_parser._add_action(action)

    # Regression Catalogue Builder
    regression_parser = subparsers.add_parser("regression", help="Build a regression-based mutation catalogue.")
    for action in parse_regression_builder()._actions:
        if action.dest != "help":
            regression_parser._add_action(action)

    args = parser.parse_args()

    # Pass `args` directly to avoid re-parsing
    if args.command == "binary":
        main_binary_builder(args)
    elif args.command == "regression":
        main_regression_builder(args)


if __name__ == "__main__":
    main()
