def edit_output_json(out_folder, config, cmd, participant_label=None):
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
        if not isinstance(sub_list, list):
            sub_list = [sub_list]
        for sub_ses_dict in sub_list:
            if "session" in sub_ses_dict.keys():
                ses, run_id = sub_ses_dict["session"], sub_ses_dict.get("sr-id", 1000)
                ses_path = f"ses-{ses}"
                output_sub_ses = out_folder / sub_path / ses_path / "anat"
                out_json = output_sub_ses / f"{sub_path}_{ses_path}_rec-SR_id-{run_id}_T2w.json"
            else:
                run_id = sub_ses_dict["sr-id"]
                output_sub_ses = out_folder / sub_path / "anat"
                out_json = output_sub_ses / f"{sub_path}_rec-SR_id-{run_id}_T2w.json"
            print(out_json)
            if not os.path.exists(out_json):
                print(f"Warning: {out_json} not found.")
                continue
            with open(out_json, "r") as f:
                mialsrtk_data = json.load(f)
            # if (
            #     "sr-id" in mialsrtk_data.keys()
            #     and "session" in mialsrtk_data.keys()
            # ):
            #     raise RuntimeError(
            #         f"The json metadata have already been modified in {out_json}. Aborting."
            #     )
            conf = {k: sub_ses_dict[k] for k in OUT_JSON_ORDER if k in sub_ses_dict.keys()}
            conf["sr-id"] = run_id
            conf["config_path"] = str(config)
            conf["Description"] = mialsrtk_data["Description"]
            conf["CustomMetaData"] = mialsrtk_data["CustomMetaData"]
            conf["Input sources run order"] = mialsrtk_data["Input sources run order"]
            custom_key = "custom_interfaces"
            custom_interfaces = (
                sub_ses_dict[custom_key] if custom_key in sub_ses_dict.keys() else {}
            )
            conf["info"] = {
                "reconstruction": "mialSRTK",
                "custom_interfaces": custom_interfaces,
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
            mask_list, auto_masks = filter_and_complement_mask_list(stacks, sub, ses, mask_list)
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
            merge_and_overwrite_folder(os.path.join(path, dirname), os.path.join(dest, dirname))
        os.rmdir(path)


def main(argv=None):
    import os
    import time
    from pathlib import Path
    from .parser import get_default_parser

    DOCKER_VERSION = "latest"

    p = get_default_parser("MIALSRTK")
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
        "--txt_to",
        default=None,
        help="Where the text output is stored. By default, it is output to the command line.",
    )
    p.add_argument(
        "--labels_derivatives_dir",
        default=None,
        help="Where the labels are stored.",
    )

    p.add_argument(
        "--atlas_dir",
        default=None,
        help="Where Gholipour's STA is stored.",
    )
    p.add_argument(
        "--pymialsrtk_path",
        default=None,
        help="Where pymialsrtk is located.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose output",
    )

    args = p.parse_args(argv)

    data_path = Path(args.data_path).absolute()
    docker_version = args.docker_version
    run_type = args.run_type
    participant_label = args.participant_label
    txt_to = args.txt_to
    param_file = args.config
    out_folder = args.out_path
    masks_derivatives_dir = Path(args.masks_path).resolve()
    atlas_dir = Path(args.atlas_dir).resolve()
    labels_derivatives_dir = args.labels_derivatives_dir
    if masks_derivatives_dir:
        masks_derivatives_dir = Path(masks_derivatives_dir).absolute()
    if labels_derivatives_dir:
        labels_derivatives_dir = Path(labels_derivatives_dir).absolute()

    pymialsrtk_path = args.pymialsrtk_path
    verbose = args.verbose

    if masks_derivatives_dir is None:
        raise RuntimeError("masks_derivatives_dir should be defined.")

    if not out_folder:
        raise RuntimeError("Please define out_folder.")
    out_folder = Path(out_folder).absolute()
    if not args.fake_run:
        os.makedirs(out_folder, exist_ok=True)

    if param_file:
        param_file = Path(param_file).absolute()
        subject_json = Path("/code/") / param_file.name

    base_command = (
        f"docker run --rm -t -u $(id -u):$(id -g)"
        f" -v {data_path}:/bids_dir"
        f" -v {out_folder}:/output_dir"
    )
    if pymialsrtk_path is not None:
        base_command += (
            f" -v {pymialsrtk_path}:/opt/conda/lib/python3.7/site-" f"packages/pymialsrtk/"
        )
    if masks_derivatives_dir is not None:
        base_command += f" -v {masks_derivatives_dir}:/masks"
    if labels_derivatives_dir is not None:
        base_command += f" -v {labels_derivatives_dir}:/labels"
    if param_file:
        base_command += f" -v {param_file.parent}:/code"
    base_command += (
        f" -v {atlas_dir}:/sta"
        f" sebastientourbier/mialsuperresolutiontoolkit-bidsapp"  # -bidsapp
        f":{docker_version}"
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
        base_command += " --masks_derivatives_dir /masks"
    if labels_derivatives_dir is not None:
        base_command += " --labels_derivatives_dir /labels"
    if txt_to:
        base_command += f" > {txt_to}"
    time_base = time.time()
    print(base_command)
    if not args.fake_run:
        os.system(base_command)
        print(f"Total elapsed time: {time.time()-time_base}")
    out_final = out_folder / f"{out_folder.name}"

    if not args.fake_run:
        # Renaming the pymialsrtk output to a folder with the same name as the output folder.
        merge_and_overwrite_folder(out_folder / f"pymialsrtk-{docker_version}", out_final)

        edit_output_json(
            out_final,
            param_file.absolute(),
            base_command,
            participant_label,
        )


if __name__ == "__main__":
    main()
