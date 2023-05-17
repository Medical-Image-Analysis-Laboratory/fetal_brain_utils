"""
Run Ebner's pipeline and parameter study on data given a reference.
The reference must be in the subject's data directory, and described as
sub-<id>_ses-<id>_desc-iso_T2w.nii.gz. The corresponding mask is given as
sub-<id>_ses-<id>_desc-iso_mask.nii.gz

Note that, currently, for simulated data, the brain extraction module
will crash for reasons that are beyond my understanding.
"""
from fetal_brain_utils import (
    iter_dir,
    get_cropped_stack_based_on_mask,
    filter_run_list,
    find_run_id,
)
from fetal_brain_utils import (
    filter_and_complement_mask_list as filter_mask_list,
)
import csv
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
    use_preprocessed,
    fake_run,
):

    pid = os.getpid()
    if participant_label:
        if sub not in participant_label:
            return
    if sub not in bids_layout.get_subjects():
        print(f"Subject {sub} not found in {data_path}")
        return

    out_suffix = "" if not use_preprocessed else "_bcorr"

    # Prepare the output path, and locate the
    # pre-computed masks
    output_path = data_path / output_path
    niftymic_out_path = output_path / "run_files"

    masks_layout = BIDSLayout(masks_folder, validate=False)
    # Pre-processing: mask (and crop) the low-resolution stacks
    cropped_path_base = niftymic_out_path / ("preprocess_ebner" + out_suffix)
    cropped_mask_path_base = niftymic_out_path / ("preprocess_mask_ebner" + out_suffix)
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

            # Construct the data and mask path from their respective
            # base paths
            input_path = data_path / sub_ses_anat
            mask_path = masks_folder / sub_ses_anat

            input_cropped_path = cropped_path_base / sub_ses_anat / run_path
            mask_cropped_path = cropped_mask_path_base / sub_ses_anat / run_path
            if not fake_run:
                os.makedirs(input_cropped_path, exist_ok=True)
                os.makedirs(mask_cropped_path, exist_ok=True)

            # Get in-plane resolution to be set as target resolution.
            resolution = 0.8 #ni.load(img_list[0]).header["pixdim"][1]

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
                if imc is not None:
                    imc = ni.Nifti1Image(imc.get_fdata() * maskc.get_fdata(), imc.affine)

                    ni.save(imc, cropped_im)
                    ni.save(maskc, cropped_mask)

                    # Define the file and path names inside the docker volume
                    run_im = Path("/data") / im_file
                    run_mask = Path("/seg") / mask_file
                    filename_data.append(str(run_im))
                    filename_masks.append(str(run_mask))
                    filename_prepro.append("/srr/preprocessing_n4itk/" + os.path.basename(run_im))
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
                f"docker run --rm -v {input_path}:/data " #               
                f"-v {mask_path}:/seg "
                f"-v {recon_path}:/srr "
                f"-v /media/paul/data/paul/fetal_uncertainty_reconstruction/reconstruction_methods/NiftyMIC/niftymic:/app/NiftyMIC/niftymic "
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

            # Creating the dataset_description file.
            os.makedirs(output_path / "niftymic", exist_ok=True)
            dataset_description = {
                "Name": "CHUV fetal brain MRI",
                "BIDSVersion": "1.8.0",
            }
            with open(output_path / "niftymic" / "dataset_description.json", "w") as f:
                json.dump(dataset_description, f)

            out_path = output_path / "niftymic" / sub_path / ses_path / "anat"
            os.makedirs(out_path, exist_ok=True)
            final_base = str(out_path / f"{sub_path}_{ses_path}_{run_path}")
            final_reject = final_base + "_desc-rejectedSlices.csv"
            final_rec = final_base + "_desc-SR_T2w.nii.gz"
            final_rec_json = final_base + "_desc-SR_T2w.json"
            final_mask = final_base + "_desc-SR_mask.nii.gz"
            final_uncertainty = final_base + "_desc-SR_uncertainty.nii.gz"
            final_uncertainty_normalized = final_base + "_desc-SR_uncertaintyNormalized.nii.gz"

            shutil.copyfile(
                recon_path / "recon_template_space/srr_template.nii.gz",
                final_rec,
            )

            shutil.copyfile(
                recon_path / "recon_template_space/srr_template_mask.nii.gz",
                final_mask,
            )
            
            shutil.copyfile(
                recon_path / "recon_subject_space/srr_rejectedSlices.csv",
                final_reject,
            )

            shutil.copyfile(
                recon_path / "recon_template_space/srr_template_mask_uncertainty.nii.gz",
                final_uncertainty,
            )

            shutil.copyfile(
                recon_path / "recon_template_space/srr_template_mask_uncertainty_normalized.nii.gz",
                final_uncertainty_normalized,
            )


            conf["info"] = {
                "reconstruction": "NiftyMIC",
                "alpha": alpha,
                "command": cmd,
            }
            conf = {k: conf[k] for k in OUT_JSON_ORDER if k in conf.keys()}
            with open(final_rec_json, "w") as f:
                json.dump(conf, f, indent=4)
        except Exception:
            msg = f"{sub_path} - {ses_path} failed:\n{traceback.format_exc()}"
            print(msg)
            failure_list.append(msg)
        if len(failure_list) > 0:
            print("SOME RUNS FAILED:")
            for e in failure_list:
                print(e)


def main():

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
        required=True,
        help="Folder where the masks are located.",
    )

    p.add_argument(
        "--out_path",
        required=True,
        help="Folder where the output will be stored.",
    )
    p.add_argument(
        "--alpha",
        help="Alpha to be used.",
        default=ALPHA,
        type=float,
    )

    p.add_argument(
        "--config",
        help="Config path.",
        required=True,
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
        help="Whether the parameter study should use " "bias corrected images as input.",
    )

    p.add_argument(
        "--fake_run",
        action="store_true",
        default=False,
        help="Whether to only print the commands instead of running them",
    )
    args = p.parse_args()
    data_path = Path(args.data_path).resolve()
    config = Path(args.config)
    masks_folder = Path(args.masks_folder).resolve()
    out_path = Path(args.out_path).resolve()
    alpha = args.alpha
    participant_label = args.participant_label
    use_preprocessed = args.use_preprocessed
    nprocs = args.nprocs
    fake_run = args.fake_run

    # Load a dictionary of subject-session-paths
    # sub_ses_dict = iter_dir(data_path, add_run_only=True)

    bids_layout = BIDSLayout(data_path, validate=False)

    with open(data_path / "code" / config, "r") as f:
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
    main()
