"""
Run the NeSVor pipeline on data.
"""
from ..utils import iter_dir, filter_and_complement_mask_list, filter_run_list
import argparse
from pathlib import Path
import os
from functools import partial
import json
import re
from bids import BIDSLayout

# Default data path
DATA_PATH = Path("/media/tsanchez/tsanchez_data/data/data")
OUT_JSON_ORDER = [
    "sr-id",
    "session",
    "ga",
    "stacks",
    "use_auto_mask",
    "config_path",
    "info",
    "im_path",
    "mask_path",
]


def list_masks(bids_layout, mask_pattern_list):
    """Given a BIDSLayout and a list of mask_patterns,
    tries to find the masks that exist for each (subject, session, run)
    in the BIDS dataset, using the provided patterns.
    """
    from fetal_brain_qc.utils import fill_pattern

    file_list = []
    for sub, ses, run, out in iter_bids(bids_layout):

        paths = [
            fill_pattern(bids_layout, sub, ses, run, p)
            for p in mask_pattern_list
        ]
        for i, f in enumerate(paths):
            if os.path.exists(f):
                fname = Path(out).name.replace(".nii.gz", "")
                file_list.append(
                    {
                        "name": fname,
                        "sub": sub,
                        "ses": ses,
                        "run": run,
                        "im": out,
                        "mask": f,
                    }
                )
                break
            if i == len(paths) - 1:
                print(
                    f"WARNING: No mask found for sub-{sub}, ses-{ses}, run-{run}"
                )
    return file_list


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


def crop_input(sub, ses, output_path, img_list, mask_list, masking=False):
    import nibabel as ni
    from fetal_brain_utils import get_cropped_stack_based_on_mask
    from functools import partial

    sub_ses_output = output_path / f"sub-{sub}/ses-{ses}/anat"
    os.makedirs(sub_ses_output, exist_ok=True)

    boundary_mm = 15
    crop_path = partial(
        get_cropped_stack_based_on_mask,
        boundary_i=boundary_mm,
        boundary_j=boundary_mm,
        boundary_k=0,
    )
    im_list_c, mask_list_c = [], []
    for image, mask in zip(img_list, mask_list):
        print(f"Processing {image} {mask}")
        im_file, mask_file = Path(image).name, Path(mask).name
        cropped_im_path = sub_ses_output / im_file
        cropped_mask_path = sub_ses_output / mask_file
        im, m = ni.load(image), ni.load(mask)

        imc = crop_path(im, m)
        maskc = crop_path(m, m)
        # Masking
        if masking:
            imc = ni.Nifti1Image(
                imc.get_fdata() * maskc.get_fdata(), imc.affine
            )
        else:
            imc = ni.Nifti1Image(imc.get_fdata(), imc.affine)

        ni.save(imc, cropped_im_path)
        ni.save(maskc, cropped_mask_path)
        im_list_c.append(str(cropped_im_path))
        mask_list_c.append(str(cropped_mask_path))
    return im_list_c, mask_list_c


def iterate_subject(
    sub,
    config_sub,
    layout,
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

    print(layout.get(subject=sub))
    if sub not in layout.get(subject=sub):
        print(f"Subject {sub} not found in {data_path}")
        return

    mask_base_path = Path(mask_base_path)
    output_path = Path(output_path)
    output_path_crop = output_path / "cropped_input"
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

        img_list, mask_list = crop_input(
            sub, ses, output_path_crop, img_list, mask_list
        )

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
                    "--batch-size 8192"
                )
            else:
                cmd = (
                    f"nesvor sample-volume "
                    f"--input-model {model} "
                    f"--output-resolution {res} "
                    f"--output-volume {output_file} "
                    f"--output-resolution {res}"
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


def main():
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

    data_path = Path(args.data_path)
    # Load a dictionary of subject-session-paths

    layout = BIDSLayout(data_path, validate=False)

    with open(data_path / "code" / args.config, "r") as f:
        params = json.load(f)
    # Iterate over all subjects and sessions
    iterate = partial(
        iterate_subject,
        layout=layout,
        data_path=data_path,
        output_path=args.out_path,
        mask_base_path=args.masks_folder,
        participant_label=args.participant_label,
        target_res=args.target_res,
        config=args.config,
    )
    for sub, config_sub in params.items():
        iterate(sub, config_sub)


if __name__ == "__main__":
    main()
