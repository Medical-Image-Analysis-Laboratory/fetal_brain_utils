"""
Run the SVRTK stacks-and-masks-selection script on a BIDS dataset.

This script runs the Quality Control (stack selection) step of SVRTK.
It takes as input a BIDS dataset, masks and a config and outputs
to out_path/svrtk/stats_summary.csv a summary of the stacks that were
excluded by SVRTK

"""
from fetal_brain_utils import (
    get_cropped_stack_based_on_mask,
    filter_run_list,
    find_run_id,
)

from fetal_brain_utils.definitions import OUT_JSON_ORDER
import argparse
from pathlib import Path
import os
import numpy as np
from functools import partial
import nibabel as ni
import json
import traceback
from bids import BIDSLayout

# Default data path
DATA_PATH = Path("/media/tsanchez/tsanchez_data/data/data")
# Default parameters for the parameter sweep
# Relative paths in the docker image to various scripts.


def filter_mask_list(stacks, sub, ses, mask_list):
    """Filter and sort a run list according to the stacks ordering"""
    run_dict = find_run_id(mask_list)
    exclude_id = []
    for s in stacks:
        if s not in run_dict.keys():
            print(f"Mask for stack {s} not found.")
            exclude_id.append(s)
    return [str(run_dict[s]) for s in stacks if s not in exclude_id], exclude_id


def iterate_subject(
    sub,
    config_sub,
    config_path,
    bids_layout,
    data_path,
    output_path,
    masks_folder,
    participant_label,
    fake_run,
):
    output_path = Path(output_path)
    if participant_label:
        if sub not in participant_label:
            return
    if sub not in bids_layout.get_subjects():
        print(f"Subject {sub} not found in {data_path}")
        return

    # Prepare the output path, and locate the
    # pre-computed masks
    # output_path = data_path / output_path
    docker_out_path = output_path / "run_files"
    recon_out_path = output_path / "svrtk"
    masks_layout = BIDSLayout(masks_folder, validate=False)
    # Pre-processing: mask (and crop) the low-resolution stacks
    cropped_path_base = docker_out_path / "preprocess"

    # Output for the parameter study.
    if not fake_run:
        os.makedirs(docker_out_path, exist_ok=True)

    # Format the directory of parameter sweep
    sub_path = f"sub-{sub}"
    if not isinstance(config_sub, list):
        config_sub = [config_sub]
    failure_list = []

    for conf in config_sub:
        try:
            ses = conf["session"] if "session" in conf else None
            img_list = bids_layout.get_runs(
                subject=sub,
                session=ses,
                extension="nii.gz",
                return_type="filename",
            )
            mask_list = masks_layout.get_runs(
                subject=sub,
                session=ses,
                extension="nii.gz",
                return_type="filename",
            )
            # stacks = conf["stacks"] if "stacks" in conf else find_run_id(img_list)
            stacks = find_run_id(img_list)
            run_id = conf["sr-id"] if "sr-id" in conf else "1"

            run_path = f"run-{run_id}"
            ses_path = f"ses-{ses}" if ses is not None else ""

            mask_list, exclude_id = filter_mask_list(stacks, sub, ses, mask_list)
            excluded_stacks = list(
                find_run_id([i for i, s in zip(img_list, stacks) if s in exclude_id]).keys()
            )
            stacks = [s for s in stacks if s not in exclude_id]
            img_list = filter_run_list(stacks, img_list)
            conf["im_path"] = img_list
            conf["mask_path"] = mask_list
            conf["config_path"] = str(config_path)
            if ses_path != "":
                sub_ses_anat = f"{sub_path}/{ses_path}/anat"
            else:
                sub_ses_anat = f"{sub_path}/anat"

            # Construct the data and mask path from their respective
            # base paths

            input_cropped_path = cropped_path_base / sub_ses_anat / run_path
            if not fake_run:
                os.makedirs(input_cropped_path, exist_ok=True)
                # os.makedirs(mask_cropped_path, exist_ok=True)

            # Get in-plane resolution to be set as target resolution.
            ip_res = []
            tp_res = []
            # Construct the path to each data point and mask in
            # the filesystem of the docker image
            filename_data, filename_masks = [], []
            boundary_mm = 15
            crop_path = partial(
                get_cropped_stack_based_on_mask,
                boundary_i=boundary_mm,
                boundary_j=boundary_mm,
                boundary_k=0,
            )
            assert len(img_list) == len(mask_list)
            for image, mask in zip(img_list, mask_list):
                print(f"Processing {image} {mask}")
                im_file = Path(image).name
                m_file = Path(mask).name
                cropped_im = input_cropped_path / im_file
                cropped_m = input_cropped_path / m_file
                im, m = ni.load(image), ni.load(mask)
                ip_res.append(im.header["pixdim"][1])
                tp_res.append(im.header["pixdim"][3])
                if not fake_run:
                    imc = crop_path(im, m)

                    maskc = crop_path(m, m)
                    imc = ni.Nifti1Image(imc.get_fdata(), imc.affine, imc.header)
                    maskc = ni.Nifti1Image(maskc.get_fdata(), imc.affine, imc.header)
                    ni.save(imc, cropped_im)
                    ni.save(maskc, cropped_m)

                # Define the file and path names inside the docker volume
                run_im = Path("/home/data") / im_file
                run_m = Path("/home/data") / m_file
                filename_data.append(str(run_im))
                filename_masks.append(str(run_m))

            nstacks = len(filename_data)
            filename_data = " ".join(filename_data)
            filename_masks = " ".join(filename_masks)
            ##
            # QC stage
            ##

            recon_path = recon_out_path / sub_ses_anat
            if not fake_run:
                os.makedirs(recon_path, exist_ok=True)

            cmd = (
                "docker run "
                f"-v {input_cropped_path}:/home/data "
                f"-v {recon_path}:/home/out/ "
                "svrtk_custom:auto-2.10 "
                f"/home/MIRTK/build/lib/tools/stacks-and-masks-selection {nstacks} "
                f"{filename_data} {filename_masks} "
            )

            print("SVRTK's QC")
            print(cmd)
            print()
            # if not fake_run:
            os.system(cmd)
            import pandas as pd

            with open(recon_path / "stats_summary.txt", "r") as file:
                data = file.read()
            data = data.replace("\n", "").split(" ")

            stacks_tot = sorted(stacks + excluded_stacks)
            excluded_stacks += find_run_id(data[2:]) if len(data) > 2 else []

            df = pd.DataFrame.from_dict(
                {
                    "sub": [sub] * len(stacks_tot),
                    "ses": [ses] * len(stacks_tot),
                    "run": stacks_tot,
                    "excluded": [int(s in excluded_stacks) for s in stacks_tot],
                }
            )
            out_final = recon_out_path / "stats_summary.csv"

            # Read the file as a panda csv

            if os.path.isfile(out_final):
                df_file = pd.read_csv(out_final)
                # Use concat to append the rows
                df_file = pd.concat([df_file, df])
                df_file.to_csv(out_final, index=False)
            else:
                df.to_csv(out_final, index=False)

        except Exception:
            msg = f"{sub_path} - {ses_path} failed:\n{traceback.format_exc()}"
            print(msg)
            failure_list.append(msg)
            return failure_list


def main(argv=None):
    from parser import get_default_parser

    p = get_default_parser("SVRTK's quality control")

    args = p.parse_args(argv)
    data_path = Path(args.data_path).resolve()
    config = Path(args.config).resolve()
    masks_folder = Path(args.masks_path).resolve()
    out_path = Path(args.out_path).resolve()
    participant_label = args.participant_label
    fake_run = args.fake_run

    bids_layout = BIDSLayout(data_path, validate=False)

    with open(config, "r") as f:
        params = json.load(f)
    # Iterate over all subjects and sessions
    iterate = partial(
        iterate_subject,
        bids_layout=bids_layout,
        config_path=config,
        data_path=data_path,
        output_path=out_path,
        masks_folder=masks_folder,
        participant_label=participant_label,
        fake_run=fake_run,
    )
    failure_list = []
    for sub, config_sub in params.items():
        out = iterate(sub, config_sub)
        failure_list += out if out is not None else []
    if len(failure_list) > 0:
        print("SOME RUNS FAILED:")
        for e in failure_list:
            print(e)


if __name__ == "__main__":
    main()
