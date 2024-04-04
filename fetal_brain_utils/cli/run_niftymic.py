"""
Run Ebner's pipeline and parameter study on data given a reference.
The reference must be in the subject's data directory, and described as
sub-<id>_ses-<id>_desc-iso_T2w.nii.gz. The corresponding mask is given as
sub-<id>_ses-<id>_desc-iso_mask.nii.gz

Note that, currently, for simulated data, the brain extraction module
will crash for reasons that are beyond my understanding.
"""

from fetal_brain_utils import (
    filter_run_list,
    find_run_id,
    crop_input,
)
from fetal_brain_utils import (
    filter_and_complement_mask_list as filter_mask_list,
)
import re
from fetal_brain_utils.definitions import OUT_JSON_ORDER
import argparse
from pathlib import Path
import os
import numpy as np
import multiprocessing
from functools import partial
import nibabel as ni
import shutil
import json
import traceback
from bids import BIDSLayout

# Default data path
DATA_PATH = Path("/media/tsanchez/tsanchez_data/data/data")
# Default parameters for the parameter sweep
ALPHA = 0.068
# Relative paths in the docker image to various scripts.
RECON_FROM_SLICES = "NiftyMIC/niftymic_reconstruct_volume_from_slices.py"
RECONSTRUCTION_PYTHON = "NiftyMIC/niftymic/application/run_reconstruction_pipeline.py"
MODIFIED_NIFTYMIC = "/home/tsanchez/Documents/mial/repositories/NiftyMIC/niftymic"


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


def iterate_subject(
    sub,
    config_sub,
    config_path,
    bids_layout,
    data_path,
    output_path,
    masks_folder,
    alpha,
    participant_label,
    automated_template,
    resolution,
    mask_input,
    fake_run,
    boundary_mm=15,
    niftymic_path=None,
    no_preprocessing=False,
):
    if participant_label:
        if sub not in participant_label:
            return
    if sub not in bids_layout.get_subjects():
        print(f"Subject {sub} not found in {data_path}")
        return

    # Prepare the output path, and locate the
    # pre-computed masks
    output_path = data_path / output_path
    niftymic_out_path = output_path / "run_files"

    masks_layout = BIDSLayout(masks_folder, validate=False)
    # Pre-processing: mask (and crop) the low-resolution stacks
    cropped_path_base = niftymic_out_path / "cropped_input"
    # Output for the reconstruction stage
    recon_path_base = niftymic_out_path / "recon_ebner"

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
            stacks = conf["stacks"] if "stacks" in conf else find_run_id(img_list)
            run_id = conf["sr-id"] if "sr-id" in conf else "1"

            run_path = f"run-{run_id}"
            ses_path = f"ses-{ses}" if ses is not None else ""

            mask_list, auto_masks = filter_mask_list(stacks, sub, ses, mask_list)
            img_list = filter_run_list(stacks, img_list)
            conf["use_auto_mask"] = auto_masks
            conf["im_path"] = img_list
            conf["mask_path"] = mask_list
            conf["config_path"] = str(config_path)
            if ses_path != "":
                sub_ses_anat = f"{sub_path}/{ses_path}/anat"
            else:
                sub_ses_anat = f"{sub_path}/anat"

            # Get in-plane resolution to be set as target resolution.
            resolution = (
                ni.load(img_list[0]).header["pixdim"][1] if resolution is None else resolution
            )

            # Construct the path to each data point and mask in
            # the filesystem of the docker image
            img_list, mask_list = crop_input(
                sub,
                ses,
                cropped_path_base,
                img_list,
                mask_list,
                mask_input,
                fake_run=False,
                boundary_ip=boundary_mm,
                boundary_tp=boundary_mm,
            )
            mount_base = Path(img_list[0]).parent
            filename_data = " ".join([str(Path("/data") / Path(im).name) for im in img_list])
            filename_masks = " ".join([str(Path("/data") / Path(m).name) for m in mask_list])
            ##
            # Reconstruction stage
            ##

            recon_path = recon_path_base / sub_ses_anat / run_path
            if not fake_run:
                os.makedirs(recon_path, exist_ok=True)

            # Replace input and mask path by preprocessed
            cmd = f"docker run -v {mount_base}:/data " f"-v {recon_path}:/srr "
            # Modified niftymic is needed for taking extra-frame-target as input.
            if niftymic_path is not None:
                cmd += f"-v {niftymic_path}:/app/NiftyMIC/niftymic "

            cmd += (
                f"renbem/niftymic python "
                f"{RECONSTRUCTION_PYTHON} "
                f"--filenames {filename_data} "
                f" --filenames-masks {filename_masks}"
                f" --dir-output /srr"
                f" --isotropic-resolution {resolution}"
                f" --suffix-mask _mask"
                f" --alpha {alpha} "
                # f" --extra-frame-target {boundary_mm}"
                # f" --boundary-stacks {boundary_mm} {boundary_mm} {boundary_mm}"
            )
            if not automated_template:
                cmd += " --automatic-target-stack 0"
            if no_preprocessing:
                cmd += (
                    " --bias-field-correction 0 "
                    "--intensity-correction 0 "
                    "--run-recon-template-space 0 "
                    "--two-step-cycles 0 "
                    "--automatic-target-stack 0"
                )
            print("RECONSTRUCTION STAGE")
            print(cmd)
            print()
            if not fake_run:
                os.system(cmd)
                ## Copy files in BIDS format

                # Creating the dataset_description file.
                os.makedirs(output_path / "niftymic", exist_ok=True)
                dataset_description = {
                    "Name": "CHUV fetal brain MRI",
                    "BIDSVersion": "1.8.0",
                }
                with open(output_path / "niftymic" / "dataset_description.json", "w") as f:
                    json.dump(dataset_description, f)

            out_path = output_path / "niftymic" / sub_path / ses_path / "anat"
            if not fake_run:
                os.makedirs(out_path, exist_ok=True)
            final_base = (
                str(out_path / f"{sub_path}_{ses_path}_{run_path}_rec-niftymic_T2w")
                if ses is not None
                else str(out_path / f"{sub_path}_{run_path}_rec-niftymic_T2w")
            )
            final_rec = final_base + ".nii.gz"
            final_rec_json = final_base + ".json"
            final_mask = final_base + "_mask.nii.gz"

            if not fake_run:
                if no_preprocessing:
                    rec_path = recon_path / "recon_subject_space/srr_subject.nii.gz"
                    mask_path = recon_path / "recon_subject_space/srr_subject_mask.nii.gz"

                else:
                    rec_path = recon_path / "recon_template_space/srr_template.nii.gz"
                    mask_path = recon_path / "recon_template_space/srr_template_mask.nii.gz"
                shutil.copyfile(
                    rec_path,
                    final_rec,
                )

                shutil.copyfile(
                    mask_path,
                    final_mask,
                )

                # Mask the output image
                im = ni.load(final_rec)
                m = ni.load(final_mask)
                im_f = im.get_fdata() * m.get_fdata()
                im = ni.Nifti1Image(im_f, im.affine, im.header)
                ni.save(im, final_rec)
                conf["info"] = {
                    "reconstruction": "NiftyMIC",
                    "alpha": alpha,
                    "command": cmd,
                }
                conf = {k: conf[k] for k in OUT_JSON_ORDER if k in conf.keys()}
                with open(final_rec_json, "w") as f:
                    json.dump(conf, f, indent=4)
        except Exception:
            msg = f"{sub} failed:\n{traceback.format_exc()}"
            print(msg)
            failure_list.append(msg)
        if len(failure_list) > 0:
            print("SOME RUNS FAILED:")
            for e in failure_list:
                print(e)


def main(argv=None):
    from .parser import get_default_parser

    p = get_default_parser("NiftyMIC")

    p.add_argument(
        "--alpha",
        help="Alpha to be used.",
        default=ALPHA,
        type=float,
    )

    p.add_argument(
        "--nprocs",
        default=1,
        help="Number of processes used in parallel",
        type=int,
    )

    p.add_argument(
        "--resolution",
        default=None,
        help="Target resolution (default = in-plane resolution)",
        type=float,
    )

    p.add_argument(
        "--no_automated_stack",
        action="store_true",
        default=False,
        help="Whether target should be selected automatically.",
    )

    p.add_argument(
        "--mask_input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether the input stacks should be masked prior to computation.",
    )

    p.add_argument(
        "--boundary_mm",
        default=15,
        type=int,
        help="Boundary added to the cropped image around the mask (in mm)",
    )

    p.add_argument(
        "--niftymic_path",
        default=None,
        help="Where the local copy of niftymic is located for mounting on the docker.",
    )

    p.add_argument(
        "--no_preprocessing",
        action="store_true",
        default=False,
        help="Whether the preprocessing should be skipped (for T2 mapping).",
    )

    args = p.parse_args(argv)
    data_path = Path(args.data_path).resolve()
    config = Path(args.config).resolve()
    masks_folder = Path(args.masks_path).resolve()
    out_path = Path(args.out_path).resolve()
    niftymic_path = Path(args.niftymic_path).resolve()
    alpha = args.alpha
    participant_label = args.participant_label
    resolution = args.resolution
    nprocs = args.nprocs
    fake_run = args.fake_run

    # Load a dictionary of subject-session-paths
    # sub_ses_dict = iter_dir(data_path, add_run_only=True)

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
        alpha=alpha,
        participant_label=participant_label,
        automated_template=not args.no_automated_stack,
        resolution=resolution,
        mask_input=args.mask_input,
        fake_run=fake_run,
        boundary_mm=args.boundary_mm,
        niftymic_path=niftymic_path,
        no_preprocessing=args.no_preprocessing,
    )
    if nprocs > 1:
        pool = multiprocessing.Pool(nprocs)

        pool.starmap(iterate, params.items())
    else:
        for sub, config_sub in params.items():
            iterate(sub, config_sub)


if __name__ == "__main__":
    main()
