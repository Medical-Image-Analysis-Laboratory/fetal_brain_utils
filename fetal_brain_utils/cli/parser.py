import argparse


def get_default_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_path",
        default=None,
        required=True,
        help="Path where the data are located",
    )
    p.add_argument(
        "--masks_path",
        default=None,
        required=True,
        help="Where the masks are stored (absolute path).",
    )

    p.add_argument(
        "--config",
        default=None,
        required=True,
        help="Where the json parameters are stored, relatively from code/ ",
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
