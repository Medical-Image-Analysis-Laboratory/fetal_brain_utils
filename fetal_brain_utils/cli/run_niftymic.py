"""
Run Ebner's pipeline and parameter study on data given a reference.
The reference must be in the subject's data directory, and described as
sub-<id>_ses-<id>_desc-iso_T2w.nii.gz. The corresponding mask is given as
sub-<id>_ses-<id>_desc-iso_mask.nii.gz

Note that, currently, for simulated data, the brain extraction module
will crash for reasons that are beyond my understanding.
"""
from mialsrtk_utils import iter_dir
import argparse
from pathlib import Path
import os
import numpy as np
import multiprocessing
from functools import partial
import nibabel as ni
from mialsrtk_utils import get_cropped_stack_based_on_mask
import json

# Default data path
DATA_PATH = Path("/media/tsanchez/tsanchez_data/data/data")
# Default parameters for the parameter sweep
ALPHAS = [
    0.001,
    0.002,
    0.005,
    0.01,
    0.013,
    0.029,
    0.068,
    0.159,
    0.369,
    0.86,
    2.0,
]  # [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
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


def filter_run_list(stacks, run_list):
    run_dict = {
        int(re.findall(r"run-(\d+)_", str(file))[-1]): file
        for file in run_list
    }
    return [v for k, v in run_dict.items() if k in stacks]


def iterate_subject(
    sub,
    config_sub,
    sub_ses_dict,
    data_path,
    output_path,
    masks_folder,
    alphas,
    participant_label,
    skip_recon,
    skip_param,
    use_preprocessed,
    use_cropped_input,
    recon_to_template_space,
    fake_run,
):

    pid = os.getpid()
    if participant_label:
        if sub not in participant_label:
            return
    if sub not in sub_ses_dict:
        print(f"Subject {sub} not found in {data_path}")
        return
    sub_dict = sub_ses_dict[sub]

    out_suffix = ""

    if not use_cropped_input:
        out_suffix += "_uncropped"
    if use_preprocessed:
        out_suffix += "_bcorr"

    # Prepare the output path, and locate the
    # pre-computed masks
    output_path = data_path / output_path
    mask_base_path = data_path / "derivatives" / masks_folder
    sub_ses_masks_dict = iter_dir(mask_base_path)

    # Pre-processing: mask (and crop) the low-resolution stacks
    cropped_path_base = output_path / ("preprocess_ebner" + out_suffix)
    cropped_mask_path_base = output_path / (
        "preprocess_mask_ebner" + out_suffix
    )
    # Output for the reconstruction stage
    recon_path_base = output_path / ("recon_ebner" + out_suffix)

    # Output for the parameter study.
    sr_study_path_base = output_path / ("recon_study_ebner" + out_suffix)
    if not fake_run:
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(sr_study_path_base, exist_ok=True)

    # Format the directory of parameter sweep
    # alphas_str = " ".join([str(a) for a in alphas])
    sub_path = f"sub-{sub}"
    if not isinstance(config_sub, list):
        config_sub = [config_sub]

    for conf in config_sub:
        if "session" not in conf:
            ses = "1"
            mask_list = sub_ses_masks_dict[sub]
            img_list = sub_ses_dict[sub]
        else:
            ses = conf["session"]
            mask_list = sub_ses_masks_dict[sub][ses]
            img_list = sub_ses_dict[sub][ses]
        stacks = conf["stacks"] if "stacks" in conf else None
        run_id = conf["sr-id"] if "sr-id" in conf else "1"
        run_path = f"run-{run_id}"
        if stacks:
            mask_list = filter_run_list(stacks, mask_list)
            img_list = filter_run_list(stacks, img_list)
        ses_path = f"ses-{ses}"

        # Construct the data and mask path from their respective
        # base paths
        input_path = data_path / sub_path / ses_path / "anat"
        mask_path = mask_base_path / sub_path / ses_path / "anat"
        if use_cropped_input:
            input_cropped_path = (
                cropped_path_base / sub_path / ses_path / "anat" / run_path
            )
            mask_cropped_path = (
                cropped_mask_path_base
                / sub_path
                / ses_path
                / "anat"
                / run_path
            )
            if not fake_run:
                os.makedirs(input_cropped_path, exist_ok=True)
                os.makedirs(mask_cropped_path, exist_ok=True)

        # Get in-plane resolution to be set as target resolution.

        resolution = ni.load(img_list[0]).header["pixdim"][1]

        # Construct the path to each data point and mask in
        # the filesystem of the docker image
        filename_data = []
        filename_masks = []
        filename_prepro = []
        for image, mask in zip(img_list, mask_list):
            if use_cropped_input and not fake_run:
                # Preprocessing run : crop and mask the low-resolution stacks
                cropped_im = str(image).replace(
                    str(input_path), str(input_cropped_path)
                )
                cropped_mask = str(mask).replace(
                    str(mask_path), str(mask_cropped_path)
                )

                im, m = ni.load(image), ni.load(mask)
                boundary_mm = 15
                imc = get_cropped_stack_based_on_mask(
                    im,
                    m,
                    boundary_i=boundary_mm,
                    boundary_j=boundary_mm,
                    boundary_k=boundary_mm,
                )

                maskc = get_cropped_stack_based_on_mask(
                    m,
                    m,
                    boundary_i=boundary_mm,
                    boundary_j=boundary_mm,
                    boundary_k=boundary_mm,
                )
                # Masking
                imc = ni.Nifti1Image(
                    imc.get_fdata() * maskc.get_fdata(), imc.affine
                )

                ni.save(imc, cropped_im)
                ni.save(maskc, cropped_mask)

            # Define the file and path names inside the docker volume
            run_im = str(image).replace(str(input_path), "/data")
            run_mask = str(mask).replace(str(mask_path), "/seg")

            filename_data.append(run_im)
            filename_masks.append(run_mask)
            filename_prepro.append(
                "/srr/preprocessing_n4itk/" + os.path.basename(run_im)
            )
        filename_data = " ".join(filename_data)
        filename_masks = " ".join(filename_masks)
        filename_prepro = " ".join(filename_prepro)
        ##
        # Reconstruction stage
        ##
        recon_path = recon_path_base / sub_path / ses_path / "anat" / run_path
        if not fake_run:
            os.makedirs(recon_path, exist_ok=True)

        # Replace input and mask path by preprocessed
        if use_cropped_input:
            input_path, mask_path = input_cropped_path, mask_cropped_path
        if not skip_recon:
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
            )
            print(f"RECONSTRUCTION STAGE (PID={pid})")
            print(cmd)
            print()
            if not fake_run:
                os.system(cmd)

        ##
        # Parameter study
        ##

        tikhonov_base = "/home/tsanchez/Documents/mial/repositories/"
        tikhonov_path = "NiftyMIC/niftymic/reconstruction/tikhonov_solver.py"
        tikhonov_local = tikhonov_base + tikhonov_path
        tikhonov_docker = "/app/" + tikhonov_path
        for alpha in alphas:
            if not skip_param:
                alpha_str = str(alpha).replace(".", "p")
                sr_study_path = (
                    sr_study_path_base
                    / sub_path
                    / ses_path
                    / f"anat/{run_path}/{alpha_str}"
                )
                if not fake_run:
                    os.makedirs(sr_study_path, exist_ok=True)

                # Path to the motion correction: obtained in the reconstruction
                # stage
                file_path = (
                    filename_prepro if use_preprocessed else filename_data
                )
                if recon_to_template_space:
                    mc_path = "/srr/recon_template_space/motion_correction/"
                    path = recon_path / "recon_template_space"
                    if not fake_run:
                        atlas, target = get_atlas_target(path)
                    else:
                        atlas, target = "lorem", "ipsum"
                else:
                    mc_path = "/srr/recon_subject_space/motion_correction/"
                output_sr = (
                    f"sub-{sub}_ses-{ses}_desc-alpha-{alpha_str}_rec-SR.nii.gz"
                )

                cmd = (
                    f"docker run -v {input_path}:/data "
                    f"-v {tikhonov_local}:{tikhonov_docker} "
                    f"-v {mask_path}:/seg "
                    f"-v {recon_path}:/srr "
                    f"-v {sr_study_path}:/srr_study "
                    f"renbem/niftymic python "
                    f"{RECON_FROM_SLICES} "
                    f"--filenames {file_path} "
                    f"--filenames-masks {filename_masks} "
                    f"--output /srr_study/{output_sr} "
                    f"--dir-input-mc {mc_path} "
                    "--reconstruction-type TK1L2 "
                    f"--alpha {alpha} "
                    f" --isotropic-resolution {resolution}"
                    f" --suffix-mask _mask"
                )
                if recon_to_template_space:
                    cmd += (
                        f" --reconstruction-space {atlas} "
                        f"--target-stack {target} "
                    )

                print(f"PARAMETER STUDY (PID={pid})")
                print(f"\tRunning alpha={alpha}")
                print(cmd)
                print()

                if not fake_run:
                    os.system(cmd)


def main(
    data_path,
    config,
    masks_folder,
    out_path,
    alphas,
    participant_label,
    skip_recon,
    skip_param,
    use_preprocessed,
    use_cropped_input,
    recon_to_template_space,
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
        data_path=data_path,
        output_path=out_path,
        masks_folder=masks_folder,
        alphas=alphas,
        participant_label=participant_label,
        skip_recon=skip_recon,
        skip_param=skip_param,
        use_preprocessed=use_preprocessed,
        use_cropped_input=use_cropped_input,
        recon_to_template_space=recon_to_template_space,
        fake_run=fake_run,
    )
    if nprocs > 1:
        pool = multiprocessing.Pool(nprocs)

        pool.starmap(iterate, params.items())
    else:
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
        "--alphas",
        help="Alphas to be tuned.",
        default=ALPHAS,
        nargs="+",
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
        "--skip_recon",
        action="store_true",
        default=False,
        help="Whether reconstruction should be skipped",
    )

    p.add_argument(
        "--skip_param",
        action="store_true",
        default=False,
        help="Whether parameter study should be skipped",
    )

    p.add_argument(
        "--use_preprocessed",
        action="store_true",
        default=False,
        help="Whether the parameter study should use "
        "bias corrected images as input.",
    )

    p.add_argument(
        "--no_cropped_input",
        action="store_false",
        dest="use_cropped_input",
        help="Whether the parameter study should *not* use "
        "cropped and masked low-resolution images as input.",
    )
    p.add_argument(
        "--use_cropped_input",
        action="store_true",
        default=True,
        help="Whether the parameter study should use "
        "cropped and masked low-resolution images as input.",
    )
    p.add_argument(
        "--recon_to_template_space",
        action="store_true",
        default=False,
        help="Whether the parameter study should use "
        "reconstructed in the template space.",
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
        alphas=args.alphas,
        participant_label=args.participant_label,
        skip_recon=args.skip_recon,
        skip_param=args.skip_param,
        use_preprocessed=args.use_preprocessed,
        use_cropped_input=args.use_cropped_input,
        recon_to_template_space=args.recon_to_template_space,
        nprocs=args.nprocs,
        fake_run=args.fake_run,
    )
