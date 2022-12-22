"""
Run the NeSVor pipeline on data.
"""
from fetal_brain_utils.utils import iter_dir, OUT_JSON_ORDER
from bids.layout.writing import build_path
import argparse
from pathlib import Path
import os
from functools import partial
import json
import re

# Default data path
DATA_PATH = Path("/media/tsanchez/tsanchez_data/data/data")
AUTO_MASK_PATH = "/media/tsanchez/tsanchez_data/data/out_anon/masks"

BATCH_SIZE = 8192


def get_mask_path(bids_dir, subject, ses, run):
    """Create the target file path from a given
    subject, run and extension.
    """
    ents = {
        "subject": subject,
        "session": ses,
        "run": run,
        "datatype": "anat",
        "acquisition": "haste",
        "suffix": "T2w_mask",
        "extension": "nii.gz",
    }

    PATTERN = (
        bids_dir + "/sub-{subject}[/ses-{session}][/{datatype}]/sub-{subject}"
        "[_ses-{session}][_acq-{acquisition}][_run-{run}]_{suffix}.{extension}"
    )
    return build_path(ents, [PATTERN])


def get_atlas_target(recon_path):
    for file in os.listdir(recon_path):
        path = os.path.join(recon_path, file)
        if ".json" in file and "reconstruct_volume_from_slices" in file:
            with open(path, "r") as f:
                config = json.load(f)
                atlas = config["reconstruction-space"]
                target = config["target-stack"]
                if "STA" in atlas:
                    break
    return atlas, target


def find_run_id(file_list):
    run_dict = {
        int(re.findall(r"run-(\d+)_", str(file))[-1]): file
        for file in file_list
    }
    return run_dict


def filter_and_complement_mask_list(stacks, sub, ses, mask_list):
    """Filter and sort a run list according to the stacks ordering"""
    run_dict = find_run_id(mask_list)
    auto_masks = []
    for s in stacks:
        if s not in run_dict.keys():
            print(f"Mask for stack {s} not found.")
            mask = get_mask_path(AUTO_MASK_PATH, sub, ses, s)
            assert os.path.isfile(mask), f"Automated mask not found at {mask}"
            print(f"Using automated mask {mask}.")
            run_dict[s] = mask
            auto_masks.append(s)
    return [run_dict[s] for s in stacks], auto_masks


def filter_run_list(stacks, run_list):
    run_dict = find_run_id(run_list)
    return [run_dict[s] for s in stacks]


def iterate_subject(
    sub,
    config_sub,
    sub_ses_dict,
    data_path,
    output_path,
    mask_base_path,
    participant_label,
    target_res,
    config,
):

    pid = os.getpid()
    if participant_label:
        if sub not in participant_label:
            return
    if sub not in sub_ses_dict:
        print(f"Subject {sub} not found in {data_path}")
        return

    mask_base_path = Path(mask_base_path)
    output_path = Path(output_path)

    sub_ses_masks_dict = iter_dir(mask_base_path)

    os.makedirs(output_path, exist_ok=True)

    sub_path = f"sub-{sub}"
    if not isinstance(config_sub, list):
        config_sub = [config_sub]

    for conf in config_sub:
        if "session" not in conf:
            ses = "01"
            mask_list = sub_ses_masks_dict[sub]
            img_list = sub_ses_dict[sub]
        else:
            ses = conf["session"]
            mask_list = sub_ses_masks_dict[sub][ses]
            img_list = sub_ses_dict[sub][ses]
        stacks = conf["stacks"] if "stacks" in conf else find_run_id(img_list)
        run_id = conf["sr-id"] if "sr-id" in conf else "1"
        run_path = f"run-{run_id}"

        mask_list, auto_masks = filter_and_complement_mask_list(
            stacks, sub, ses, mask_list
        )
        mask_list = [str(f) for f in mask_list]
        img_list = [str(f) for f in filter_run_list(stacks, img_list)]
        conf["use_auto_mask"] = auto_masks
        conf["im_path"] = img_list
        conf["mask_path"] = mask_list
        conf["config_path"] = str(config)
        ses_path = f"ses-{ses}"
        # Construct the data and mask path from their respective
        # base paths
        output_sub_ses = output_path / sub_path / ses_path / "anat"
        os.makedirs(output_sub_ses, exist_ok=True)
        # Get in-plane resolution to be set as target resolution.

        img_str = " ".join([str(im) for im in img_list])
        mask_str = " ".join([str(m) for m in mask_list])

        model = output_sub_ses / f"{sub_path}_{ses_path}_{run_path}_model.pt"

        for i, res in enumerate(target_res):
            res_str = str(res).replace(".", "p")
            output_str = (
                output_sub_ses / f"{sub_path}_{ses_path}_"
                f"acq-haste_res-{res_str}_{run_path}_T2w"
            )
            output_file = str(output_str) + ".nii.gz"
            output_json = str(output_str) + ".json"
            if i == 0:
                cmd = (
                    f"nesvor reconstruct "
                    f"--input-stacks {img_str} "
                    f"--stack-masks {mask_str} "
                    f"--output-volume {output_file} "
                    f"--output-resolution {res} "
                    f"--output-model {model} "
                    f"--batch-size {BATCH_SIZE}"
                )
            else:
                cmd = (
                    f"nesvor sample-volume "
                    f"--input-model {model} "
                    f"--output-resolution {res} "
                    f"--output-volume {output_file} "
                    f"--output-resolution {res} "
                    f"--inference-batch-size 16384"
                )
            conf["info"] = {
                "reconstruction": "NeSVoR",
                "res": res,
                "model": str(model),
                "command": cmd,
            }
            conf = {k: conf[k] for k in OUT_JSON_ORDER}

            with open(output_json, "w") as f:
                json.dump(conf, f, indent=4)
            print(cmd)
            os.system(cmd)


def main(
    data_path,
    config,
    mask_base_path,
    out_path,
    participant_label,
    target_res,
):
    # Load a dictionary of subject-session-paths
    sub_ses_dict = iter_dir(data_path, add_run_only=True)

    with open(data_path / "code" / config, "r") as f:
        params = json.load(f)
    # Iterate over all subjects and sessions
    iterate = partial(
        iterate_subject,
        sub_ses_dict=sub_ses_dict,
        data_path=data_path,
        output_path=out_path,
        mask_base_path=mask_base_path,
        participant_label=participant_label,
        target_res=target_res,
        config=config,
    )
    for sub, config_sub in params.items():
        iterate(sub, config_sub)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_path",
        default=DATA_PATH,
        help="Path where the data are located",
    )

    p.add_argument(
        "--masks_folder",
        required=True,
        default=None,
        help="Folder where the masks are located.",
    )

    p.add_argument(
        "--out_path",
        required=True,
        default=None,
        help="Folder where the output will be stored.",
    )

    p.add_argument(
        "--config",
        help="Config path in data_path/code (default: `params.json`)",
        default="params.json",
        type=str,
    )

    p.add_argument(
        "--participant_label",
        default=None,
        help="Label of the participant",
        nargs="+",
    )

    p.add_argument(
        "--target_res",
        required=True,
        nargs="+",
        type=float,
        help="Target resolutions at which the reconstruction should be done.",
    )
    args = p.parse_args()

    main(
        data_path=Path(args.data_path),
        config=args.config,
        mask_base_path=args.masks_folder,
        out_path=args.out_path,
        participant_label=args.participant_label,
        target_res=args.target_res,
    )
