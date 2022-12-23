def main():

    import os
    import time
    from pathlib import Path
    import argparse
    import sys

    DOCKER_VERSION = "v2.1.0-dev"
    OUTPUT_BASE = Path("/media/tsanchez/tsanchez_data/data/derivatives")
    PYMIALSRTK_PATH = (
        "/home/tsanchez/Documents/mial/"
        "repositories/mialsuperresolutiontoolkit/pymialsrtk"
    )
    JSON_BASE = Path("/bids_dir/code/")
    PATH_TO_ATLAS = "/media/tsanchez/tsanchez_data/data/atlas"
    DATA_PATH = Path("/media/tsanchez/tsanchez_data/data/data")

    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_path",
        default=DATA_PATH,
        help="Path where the data are located",
    )
    p.add_argument(
        "--docker_version",
        default=DOCKER_VERSION,
        help="Docker version of the pipeline used.",
    )
    p.add_argument(
        "--run_type",
        help="Type of pipeline that is run. Can choose between "
        "running the super-resolution pipeline (`sr`) "
        "or only preprocesing (`preprocessing`).",
        choices=["sr", "preprocessing"],
        default="sr",
    )
    p.add_argument(
        "--automated",
        action="store_true",
        default=False,
        help="Run with automated masks",
    )
    p.add_argument(
        "--participant_label",
        help="The label(s) of the participant(s) that should be analyzed.",
        nargs="+",
    )
    p.add_argument(
        "--txt_to",
        default=None,
        help="Where the text output is stored. By default, it is output to "
        "the command line.",
    )
    p.add_argument(
        "--param_file",
        default=None,
        help="Where the json parameters are stored, relatively from code/ ",
    )
    p.add_argument(
        "--out_folder", default=None, help="Where the results are stored."
    )
    p.add_argument(
        "--masks_derivatives_dir",
        default=None,
        help="Where the masks are stored (absolute path).",
    )
    p.add_argument(
        "--labels_derivatives_dir",
        default=None,
        help="Where the labels are stored (absolute path).",
    )
    p.add_argument(
        "--pymialsrtk_path",
        default=PYMIALSRTK_PATH,
        help="Where pymialsrtk is located.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose output",
    )
    p.add_argument(
        "--no_python_mount",
        action="store_true",
        default=False,
        help="Whether the python folder should not be mounted.",
    )
    args = p.parse_args()

    data_path = Path(args.data_path)
    docker_version = args.docker_version
    run_type = args.run_type
    participant_label = args.participant_label
    automated = args.automated
    txt_to = args.txt_to
    param_file = args.param_file
    out_folder = args.out_folder
    masks_derivatives_dir = args.masks_derivatives_dir
    labels_derivatives_dir = args.labels_derivatives_dir
    pymialsrtk_path = args.pymialsrtk_path
    verbose = args.verbose
    no_python_mount = args.no_python_mount

    mask_str = "automated" if automated else "manual"
    if not masks_derivatives_dir and not automated:
        raise RuntimeError(
            "masks_derivatives_dir should be defined when"
            " using manual masks."
        )
    elif masks_derivatives_dir and automated:
        raise RuntimeError(
            "Incompatible value of parameters automated and "
            "masks_derivatives_dir. The masks_derivatives_dir should only be "
            "defined when automated=False."
        )
    masks_derivatives_dir = None if automated else masks_derivatives_dir
    if not out_folder:
        out_folder = f"derivatives/{run_type}_{mask_str}"
        out_folder = data_path / out_folder
    os.makedirs(out_folder, exist_ok=True)

    if param_file:
        subject_json = JSON_BASE / param_file
    else:
        if automated:
            subject_json = JSON_BASE / "automated_preprocessing_config.json"
        else:
            subject_json = JSON_BASE / "manual_preprocessing_config.json"

    base_command = (
        f"docker run --rm -t -u $(id -u):$(id -g)"
        f" -v {data_path}:/bids_dir"
        f" -v {out_folder}:/output_dir"
    )
    if not no_python_mount:
        base_command += (
            f" -v {pymialsrtk_path}:/opt/conda/lib/python3.7/site-"
            f"packages/pymialsrtk/"
        )
    if masks_derivatives_dir is not None:
        base_command += f" -v {masks_derivatives_dir}:/masks"
    if labels_derivatives_dir is not None:
        base_command += f" -v {labels_derivatives_dir}:/labels"
    base_command += (
        f" -v {PATH_TO_ATLAS}:/sta"
        f" sebastientourbier/mialsuperresolutiontoolkit-"
        f"bidsapp:{docker_version}"
        f" /bids_dir /output_dir participant"
        f" --param_file {subject_json}"
        f" --openmp_nb_of_cores 3"
        f" --nipype_nb_of_cores 1"
    )
    if run_type:
        base_command += f" --run_type {run_type}"
    if participant_label:
        participant_label = " ".join(participant_label)
        base_command += f" --participant_label {participant_label}"
        # f" -it --entrypoint /bin/bash"
    if verbose:
        base_command += " --verbose"
    if masks_derivatives_dir is not None:
        base_command += f" --masks_derivatives_dir /masks"
    if labels_derivatives_dir is not None:
        base_command += f" --labels_derivatives_dir /labels"
    if txt_to:
        base_command += f" > {txt_to}"
    time_base = time.time()
    print(base_command)
    os.system(base_command)
    print(f"Total elapsed time: {time.time()-time_base}")


if __name__ == "__main__":
    main()
