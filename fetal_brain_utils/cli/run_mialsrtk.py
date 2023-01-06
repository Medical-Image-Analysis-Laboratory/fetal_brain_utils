def edit_output_json(
    out_folder, config, cmd, auto_dict, participant_label=None
):
    import json
    import os

    OUT_JSON_ORDER = ["sr-id", "session", "ga", "stacks"]

    with open(config, "r") as f:
        config_data = json.load(f)
    for sub, sub_list in config_data.items():
        sub_path = f"sub-{sub}"
        if participant_label:
            if sub not in participant_label:
                continue
        for sub_ses_dict in sub_list:

            ses, run_id = sub_ses_dict["session"], sub_ses_dict["sr-id"]
            ses_path = f"ses-{ses}"
            output_sub_ses = out_folder / sub_path / ses_path / "anat"
            out_json = (
                output_sub_ses
                / f"{sub_path}_{ses_path}_rec-SR_id-{run_id}_T2w.json"
            )
            if not os.path.exists(out_json):
                print(f"Warning: {out_json} not found.")
                continue
            with open(out_json, "r") as f:
                mialsrtk_data = json.load(f)
            if (
                "sr-id" in mialsrtk_data.keys()
                and "session" in mialsrtk_data.keys()
            ):
                raise RuntimeError(
                    f"The json metadata have already been modified in {out_json}. Aborting."
                )
            conf = {
                k: sub_ses_dict[k]
                for k in OUT_JSON_ORDER
                if k in sub_ses_dict.keys()
            }
            conf["sr-id"] = run_id
            conf["config_path"] = str(config)
            if auto_dict:
                conf["use_auto_mask"] = auto_dict[sub][ses]
            custom_key = "custom_interfaces"
            custom_interfaces = (
                sub_ses_dict[custom_key]
                if custom_key in sub_ses_dict.keys()
                else {}
            )
            conf["info"] = {
                "reconstruction": "mialSRTK",
                "desc": mialsrtk_data["Description"],
                "recon_data": mialsrtk_data["CustomMetaData"],
                "custom_interfaces": custom_interfaces,
                "stacks_ordered": mialsrtk_data["Input sources run order"],
                "command": cmd,
            }
            with open(out_json, "w") as f:
                json.dump(conf, f, indent=4)


def find_and_copy_masks(config, masks_src, masks_dest):
    import json
    from fetal_brain_utils import filter_and_complement_mask_list
    from fetal_brain_utils import iter_dir
    from collections import defaultdict
    import os
    import shutil
    from pathlib import Path

    masks_dict = iter_dir(masks_src)
    auto_dict = defaultdict(dict)
    with open(config, "r") as f:
        config_data = json.load(f)
    for sub, sub_list in config_data.items():
        sub_path = f"sub-{sub}"
        for sub_ses_dict in sub_list:
            stacks = sub_ses_dict["stacks"]
            ses = sub_ses_dict["session"]
            ses_path = f"ses-{ses}"
            mask_list = masks_dict[sub][ses]
            mask_list, auto_masks = filter_and_complement_mask_list(
                stacks, sub, ses, mask_list
            )
            auto_dict[sub][ses] = auto_masks
            output_sub_ses = Path(masks_dest / sub_path / ses_path / "anat")
            os.makedirs(output_sub_ses, exist_ok=True)
            for m in mask_list:
                shutil.copy(m, output_sub_ses / Path(m).name)
    return auto_dict


def merge_and_overwrite_folder(src, dest):
    """
    Based on https://stackoverflow.com/questions/22588225/how-do-you-merge-two-directories-or-move-with-replace-from-the-windows-command
    Moves files recursively from a src folder to a dest folder, overwriting existing
    files and deleting empty directory after files have been moved.
    """
    import os
    import shutil
    import glob

    os.makedirs(dest, exist_ok=True)

    for path, dirs, files in os.walk(src):
        relPath = os.path.relpath(path, src)
        destPath = os.path.join(dest, relPath)
        os.makedirs(destPath, exist_ok=True)
        for file in files:
            shutil.move(os.path.join(path, file), os.path.join(destPath, file))
        for dirname in dirs:
            merge_and_overwrite_folder(
                os.path.join(path, dirname), os.path.join(dest, dirname)
            )
        os.rmdir(path)


def main():

    import os
    import time
    from pathlib import Path
    import argparse
    import sys

    OUTPUT_BASE = Path("/media/tsanchez/tsanchez_data/data/derivatives")
    PYMIALSRTK_PATH = (
        "/home/tsanchez/Documents/mial/"
        "repositories/mialsuperresolutiontoolkit/pymialsrtk"
    )
    DOCKER_VERSION = "v2.1.0-dev"

    JSON_BASE = Path("/code/")
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

    p.add_argument(
        "--complement_missing_masks",
        action="store_true",
        default=False,
        help="Whether missing masks should be replaced with automated masks.",
    )
    args = p.parse_args()

    data_path = Path(args.data_path).absolute()
    docker_version = args.docker_version
    run_type = args.run_type
    participant_label = args.participant_label
    automated = args.automated
    txt_to = args.txt_to
    param_file = args.param_file
    out_folder = args.out_folder
    masks_derivatives_dir = args.masks_derivatives_dir
    labels_derivatives_dir = args.labels_derivatives_dir
    complement_missing_masks = args.complement_missing_masks
    if masks_derivatives_dir:
        masks_derivatives_dir = Path(masks_derivatives_dir).absolute()
    if labels_derivatives_dir:
        labels_derivatives_dir = Path(labels_derivatives_dir).absolute()

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
    out_folder = Path(out_folder).absolute()
    os.makedirs(out_folder, exist_ok=True)
    auto_dict = None
    if complement_missing_masks:
        assert (
            masks_derivatives_dir is not None,
            "Cannot use --complement_missing_masks if masks_derivatives_dir is None",
        )

        auto_dict = find_and_copy_masks(
            param_file,
            masks_derivatives_dir,
            out_folder / "masks",
        )
        masks_derivatives_dir = out_folder / "masks"
        print(auto_dict)

    if param_file:
        param_file = Path(param_file).absolute()
        subject_json = Path("/code/") / param_file.name
    else:
        if automated:
            subject_json = (
                Path("/bids_dir/code/") / "automated_preprocessing_config.json"
            )
        else:
            subject_json = (
                Path("/bids_dir/code/") / "manual_preprocessing_config.json"
            )

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
    if param_file:
        print(param_file.parent.absolute())
        base_command += f" -v {param_file.parent}:/code"
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
    out_final = out_folder / f"{out_folder.name}"

    # Renaming the pymialsrtk output to a folder with the same name as the output folder.
    merge_and_overwrite_folder(
        out_folder / f"pymialsrtk-{DOCKER_VERSION[1:]}", out_final
    )

    edit_output_json(
        out_final,
        param_file.absolute(),
        base_command,
        auto_dict,
        participant_label,
    )


if __name__ == "__main__":
    main()
