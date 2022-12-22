"""
Run Ebner's pipeline and parameter study on data given a reference.
The reference must be in the subject's data directory, and described as
sub-<id>_ses-<id>_desc-iso_T2w.nii.gz. The corresponding mask is given as
sub-<id>_ses-<id>_desc-iso_mask.nii.gz

Note that, currently, for simulated data, the brain extraction module
will crash for reasons that are beyond my understanding.
"""
from fetal_brain_utils.utils import (
    iter_dir,
    get_cropped_stack_based_on_mask,
    filter_run_list,
    find_run_id,
    OUT_JSON_ORDER,
)
from fetal_brain_utils.utils import (
    filter_and_complement_mask_list as filter_mask_list,
)
import argparse
from pathlib import Path
import os
import numpy as np
import multiprocessing
from functools import partial
import nibabel as ni
import shutil
import json

# Default data path
DATA_PATH = Path("/media/tsanchez/tsanchez_data/data/data")
# Default parameters for the parameter sweep
ALPHA = 0.068
# Relative paths in the docker image to various scripts.
RECON_FROM_SLICES = "NiftyMIC/niftymic_reconstruct_volume_from_slices.py"
RECONSTRUCTION_PYTHON = (
    "NiftyMIC/niftymic/application/run_reconstruction_pipeline.py"
)


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


import re


def iterate_subject(
    sub,
    config_sub,
    config_path,
    sub_ses_dict,
    data_path,
    output_path,
    masks_folder,
    alpha,
    participant_label,
    use_preprocessed,
    fake_run,
):

    pid = os.getpid()
    if participant_label:
        if sub not in participant_label:
            return
    if sub not in sub_ses_dict:
        print(f"Subject {sub} not found in {data_path}")
        return

    out_suffix = "" if not use_preprocessed else "_bcorr"

    # Prepare the output path, and locate the
    # pre-computed masks
    output_path = data_path / output_path
    niftymic_out_path = output_path / "niftymic"
    mask_base_path = data_path / "derivatives" / masks_folder

    sub_ses_masks_dict = iter_dir(mask_base_path)

    # Pre-processing: mask (and crop) the low-resolution stacks
    cropped_path_base = niftymic_out_path / ("preprocess_ebner" + out_suffix)
    cropped_mask_path_base = niftymic_out_path / (
        "preprocess_mask_ebner" + out_suffix
    )
    # Output for the reconstruction stage
    recon_path_base = niftymic_out_path / ("recon_ebner" + out_suffix)

    # Output for the parameter study.
    if not fake_run:
        os.makedirs(niftymic_out_path, exist_ok=True)

    # Format the directory of parameter sweep
    # alphas_str = " ".join([str(a) for a in alphas])
    sub_path = f"sub-{sub}"
    if not isinstance(config_sub, list):
        config_sub = [config_sub]
    failure_list = []
    for conf in config_sub:
        try:
            if "session" not in conf:
                ses = "1"
                img_list = sub_ses_dict[sub]
                mask_list = sub_ses_masks_dict[sub]

            else:
                ses = conf["session"]
                img_list = sub_ses_dict[sub][ses]
                mask_list = sub_ses_masks_dict[sub][ses]

            stacks = (
                conf["stacks"] if "stacks" in conf else find_run_id(img_list)
            )
            run_id = conf["sr-id"] if "sr-id" in conf else "1"

            run_path = f"run-{run_id}"
            ses_path = f"ses-{ses}"

            mask_list, auto_masks = filter_mask_list(
                stacks, sub, ses, mask_list
            )
            img_list = filter_run_list(stacks, img_list)
            conf["use_auto_mask"] = auto_masks
            conf["im_path"] = img_list
            conf["mask_path"] = mask_list
            conf["config_path"] = str(config_path)
            sub_ses_anat = f"{sub_path}/{ses_path}/anat"

            # Construct the data and mask path from their respective
            # base paths
            input_path = data_path / sub_ses_anat
            mask_path = mask_base_path / sub_ses_anat

            input_cropped_path = cropped_path_base / sub_ses_anat / run_path
            mask_cropped_path = (
                cropped_mask_path_base / sub_ses_anat / run_path
            )
            if not fake_run:
                os.makedirs(input_cropped_path, exist_ok=True)
                os.makedirs(mask_cropped_path, exist_ok=True)

            # Get in-plane resolution to be set as target resolution.
            resolution = ni.load(img_list[0]).header["pixdim"][1]

            # Construct the path to each data point and mask in
            # the filesystem of the docker image
            filename_data, filename_masks, filename_prepro = [], [], []
            boundary_mm = 15
            crop_path = partial(
                get_cropped_stack_based_on_mask,
                boundary_i=boundary_mm,
                boundary_j=boundary_mm,
                boundary_k=0,
            )
            for image, mask in zip(img_list, mask_list):
                print(f"Processing {image} {mask}")
                im_file, mask_file = Path(image).name, Path(mask).name
                cropped_im = input_cropped_path / im_file
                cropped_mask = mask_cropped_path / mask_file
                im, m = ni.load(image), ni.load(mask)

                imc = crop_path(im, m)
                maskc = crop_path(m, m)
                # Masking
                imc = ni.Nifti1Image(
                    imc.get_fdata() * maskc.get_fdata(), imc.affine
                )

                ni.save(imc, cropped_im)
                ni.save(maskc, cropped_mask)

                # Define the file and path names inside the docker volume
                run_im = Path("/data") / im_file
                run_mask = Path("/seg") / mask_file
                filename_data.append(str(run_im))
                filename_masks.append(str(run_mask))
                filename_prepro.append(
                    "/srr/preprocessing_n4itk/" + os.path.basename(run_im)
                )
            filename_data = " ".join(filename_data)
            filename_masks = " ".join(filename_masks)
            filename_prepro = " ".join(filename_prepro)
            ##
            # Reconstruction stage
            ##

            recon_path = recon_path_base / sub_ses_anat / run_path
            if not fake_run:
                os.makedirs(recon_path, exist_ok=True)

            # Replace input and mask path by preprocessed
            input_path, mask_path = input_cropped_path, mask_cropped_path
            cmd = (
                f"docker run -v {input_path}:/data "
                f"-v {mask_path}:/seg "
                f"-v {recon_path}:/srr "
                f"renbem/niftymic python "
                f"{RECONSTRUCTION_PYTHON} "
                f"--filenames {filename_data} "
                f" --filenames-masks {filename_masks}"
                f" --dir-output /srr"
                f" --isotropic-resolution {resolution}"
                f" --suffix-mask _mask"
                f" --alpha {alpha}"
            )
            print(f"RECONSTRUCTION STAGE (PID={pid})")
            print(cmd)
            print()
            if not fake_run:
                os.system(cmd)
            ## Copy files in BIDS format

            out_path = output_path / sub_path / ses_path / "anat"
            os.makedirs(out_path, exist_ok=True)
            final_base = str(
                out_path / f"{sub_path}_{ses_path}_{run_path}_SR_T2w"
            )
            final_rec = final_base + ".nii.gz"
            final_rec_json = final_base + ".json"
            final_mask = final_base + "_mask.nii.gz"

            shutil.copyfile(
                recon_path / "recon_template_space/srr_template.nii.gz",
                final_rec,
            )
            shutil.copyfile(
                recon_path / "recon_template_space/srr_template_mask.nii.gz",
                final_mask,
            )

            conf["info"] = {
                "reconstruction": "NiftyMIC",
                "alpha": alpha,
                "command": cmd,
            }
            conf = {k: conf[k] for k in OUT_JSON_ORDER}
            with open(final_rec_json, "w") as f:
                json.dump(conf, f, indent=4)
        except Exception as e:
            msg = f"{sub_path} - {ses_path} failed: {e}"
            print(msg)
            failure_list.append(msg)
        if len(failure_list) > 0:
            print("SOME RUNS FAILED:")
            for e in failure_list:
                print(e)


def main(
    data_path,
    config,
    masks_folder,
    out_path,
    alpha,
    participant_label,
    use_preprocessed,
    nprocs,
    fake_run,
):
    # Load a dictionary of subject-session-paths
    sub_ses_dict = iter_dir(data_path, add_run_only=True)

    with open(data_path / "code" / config, "r") as f:
        params = json.load(f)
    # Iterate over all subjects and sessions
    iterate = partial(
        iterate_subject,
        sub_ses_dict=sub_ses_dict,
        config_path=config,
        data_path=data_path,
        output_path=out_path,
        masks_folder=masks_folder,
        alpha=alpha,
        participant_label=participant_label,
        use_preprocessed=use_preprocessed,
        fake_run=fake_run,
    )
    if nprocs > 1:
        pool = multiprocessing.Pool(nprocs)

        pool.starmap(iterate, params.items())
    else:
        for sub, config_sub in params.items():
            iterate(sub, config_sub)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_path",
        default=DATA_PATH,
        help="Path where the data are located",
    )

    p.add_argument(
        "--masks_folder",
        default="masks",
        help="Folder where the masks are located. "
        "Relative to `<data_path>/derivatives` (default: `masks`)",
    )

    p.add_argument(
        "--out_path",
        default="derivatives",
        help="Folder where the output will be stored. "
        "Relative to `<data_path>` (default: `derivatives`)",
    )
    p.add_argument(
        "--alpha",
        help="Alpha to be used.",
        default=ALPHA,
        type=float,
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
        "--nprocs",
        default=1,
        help="Number of processes used in parallel",
        type=int,
    )

    p.add_argument(
        "--use_preprocessed",
        action="store_true",
        default=False,
        help="Whether the parameter study should use "
        "bias corrected images as input.",
    )

    p.add_argument(
        "--fake_run",
        action="store_true",
        default=False,
        help="Whether to only print the commands instead of running them",
    )
    args = p.parse_args()

    main(
        data_path=Path(args.data_path),
        config=args.config,
        masks_folder=args.masks_folder,
        out_path=args.out_path,
        alpha=args.alpha,
        participant_label=args.participant_label,
        use_preprocessed=args.use_preprocessed,
        nprocs=args.nprocs,
        fake_run=args.fake_run,
    )
