import argparse


def get_default_parser(srr):
    """Create a default parser with common inputs between the different reconstruction wrappers."""
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Parser for wrapper script of {srr}\n This requires to provide data in a BIDS format.",
    )
    p.add_argument(
        "--data_path",
        default=None,
        required=True,
        help="Path to the data.",
    )
    p.add_argument(
        "--masks_path",
        default=None,
        required=True,
        help="Path to the brain masks.",
    )

    p.add_argument(
        "--config",
        default=None,
        required=True,
        help="Path to the configuration file.",
    )
    p.add_argument("--out_path", default=None, required=True, help="Where the results are stored.")
    p.add_argument(
        "--participant_label",
        help="The label(s) of the participant(s) that should be analyzed.",
        nargs="+",
    )

    p.add_argument(
        "--fake_run",
        action="store_true",
        default=False,
        help="Whether to only print the commands instead of running them",
    )

    return p
