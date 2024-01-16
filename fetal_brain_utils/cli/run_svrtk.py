"""
Run Ebner's pipeline and parameter study on data given a reference.
The reference must be in the subject's data directory, and described as
sub-<id>_ses-<id>_desc-iso_T2w.nii.gz. The corresponding mask is given as
sub-<id>_ses-<id>_desc-iso_mask.nii.gz

Note that, currently, for simulated data, the brain extraction module
will crash for reasons that are beyond my understanding.
"""
from fetal_brain_utils import (
    get_cropped_stack_based_on_mask,
    filter_run_list,
    find_run_id,
)
from fetal_brain_utils import (
    filter_and_complement_mask_list as filter_mask_list,
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
import shutil

# Default data path
DATA_PATH = Path("/media/tsanchez/tsanchez_data/data/data")
SVRTK_AUTO_VERSION = "custom_2.20"
# Default parameters for the parameter sweep
# Relative paths in the docker image to various scripts.


def iterate_subject(
    sub,
    config_sub,
    config_path,
    bids_layout,
    data_path,
    output_path,
    target_res,
    masks_folder,
    participant_label,
    use_svrtk_auto,
    fake_run,
):
    output_path = Path(output_path)
    if participant_label:
        if sub not in participant_label:
            return
    if sub not in bids_layout.get_subjects():
        print(f"Subject {sub} not found in {data_path}")
        return

    # Prepare the output path
    docker_out_path = output_path / "run_files"
    recon_out_path = output_path / "svrtk"
    prepro_path_base = docker_out_path / "preprocess"
    recon_path_base = docker_out_path / "recon"
    if not fake_run:
        os.makedirs(docker_out_path, exist_ok=True)

    # Format the directory of parameter sweep
    sub_path = f"sub-{sub}"
    if not isinstance(config_sub, list):
        config_sub = [config_sub]
    failure_list = []

    if not use_svrtk_auto:
        masks_layout = BIDSLayout(masks_folder, validate=False)
    for conf in config_sub:
        try:
            ses = conf["session"] if "session" in conf else None
            img_list = bids_layout.get_runs(
                subject=sub,
                session=ses,
                extension="nii.gz",
                return_type="filename",
            )
            stacks = conf["stacks"] if "stacks" in conf else find_run_id(img_list)
            run_id = conf["sr-id"] if "sr-id" in conf else "1"

            run_path = f"run-{run_id}"
            ses_path = f"ses-{ses}" if ses is not None else ""

            img_list = filter_run_list(stacks, img_list)
            conf["im_path"] = img_list
            if not use_svrtk_auto:
                mask_list = masks_layout.get_runs(
                    subject=sub,
                    session=ses,
                    extension="nii.gz",
                    return_type="filename",
                )
            else:
                mask_list = None

                conf["mask_path"] = mask_list
            conf["config_path"] = str(config_path)
            if ses_path != "":
                sub_ses_anat = f"{sub_path}/{ses_path}/anat"
            else:
                sub_ses_anat = f"{sub_path}/anat"

            # Construct the data and mask path from their respective
            # base paths
            input_path = data_path / sub_ses_anat

            input_prepro_path = prepro_path_base / sub_ses_anat / run_path
            if not fake_run:
                os.makedirs(input_prepro_path, exist_ok=True)
                # os.makedirs(mask_cropped_path, exist_ok=True)

            # Get in-plane resolution to be set as target resolution.
            tp_res = []
            # Construct the path to each data point and mask in
            # the filesystem of the docker image
            # filename_data, filename_masks = [], []
            filename_data = []
            boundary_mm = 15
            crop_path = partial(
                get_cropped_stack_based_on_mask,
                boundary_i=boundary_mm,
                boundary_j=boundary_mm,
                boundary_k=0,
            )
            it_list = zip(img_list, mask_list) if not use_svrtk_auto else img_list
            for o in it_list:  # mask_list
                image = o if use_svrtk_auto else o[0]
                mask = None if use_svrtk_auto else o[1]
                # Copy the image file to input_prepro_path
                im_file = Path(image).name
                prepro_im = input_prepro_path / im_file
                im = ni.load(image)
                tp_res.append(round(im.header["pixdim"][3], 1))

                if use_svrtk_auto:
                    if not fake_run:
                        shutil.copy(image, prepro_im)
                else:
                    print(f"Processing {image} {mask}")
                    m = ni.load(mask)
                    if not fake_run:
                        imc = crop_path(im, m)
                        imc = ni.Nifti1Image(imc.get_fdata(), imc.affine)
                        ni.save(imc, prepro_im)

                # Define the file and path names inside the docker volume
                run_im = Path("/home/data") / im_file
                filename_data.append(str(run_im))
            filename_data = " ".join(filename_data)
            # filename_masks = " ".join(filename_masks)
            tp = str(round(np.mean(tp_res), 2))
            tp_str = " ".join([str(t) for t in tp_res])
            ##
            # Reconstruction stage
            ##
            recon_path = recon_path_base / sub_ses_anat
            if not fake_run:
                os.makedirs(recon_path, exist_ok=True)

            # Replace input and mask path by preprocessed
            input_path = input_prepro_path
            # , mask_path = , mask_cropped_path
            recon_file = f"{sub_path}_{ses_path}_{run_path}_rec-SR_T2w.nii.gz"

            if use_svrtk_auto:
                cmd = (
                    "docker run "
                    f"-v {input_path}:/home/data "
                    f"-v {recon_path}:/home/out/ "
                    f"fetalsvrtk/svrtk:{SVRTK_AUTO_VERSION} "
                    "bash /home/auto-proc-svrtk/auto-brain-reconstruction.sh "
                    f"/home/data /home/out/ 1 {tp} {target_res} 1"
                )
            else:
                cmd = (
                    "docker run "
                    f"-v {input_path}:/home/data "
                    # f"-v {mask_path}:/home/mask "
                    f"-v {recon_path}:/home/out/ "
                    "fetalsvrtk/svrtk mirtk reconstruct "
                    f"/home/out/{recon_file} {len(img_list)} "
                    f"{filename_data} "
                    f"-thickness {tp_str} -resolution {target_res}"
                )

            print("RECONSTRUCTION STAGE")
            print(cmd)
            print()
            if not fake_run:
                os.system(cmd)
                # Copy files in BIDS format
                recon_out = recon_out_path / sub_ses_anat
                os.makedirs(recon_out, exist_ok=True)

                shutil.copy(recon_path / "reo-SVR-output-brain.nii.gz", recon_out / recon_file)

                # Creating the dataset_description file.
                os.makedirs(output_path / "svrtk", exist_ok=True)
                dataset_description = {
                    "Name": "SVRTK fetal brain MRI reconstruction",
                    "BIDSVersion": "1.8.0",
                }
                with open(output_path / "svrtk" / "dataset_description.json", "w") as f:
                    json.dump(dataset_description, f)

                final_rec_json = recon_out / recon_file.replace("nii.gz", "json")

                conf["info"] = {
                    "reconstruction": "SVRTK",
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


def main(argv=None):
    from .parser import get_default_parser

    p = get_default_parser("SVRTK")
    p.add_argument(
        "--use_svrtk_auto",
        action="store_true",
        help="Use svrtk:auto-2.20 instead of svrtk:latest. This features automated reorientation.",
    )
    p.add_argument(
        "--target_res",
        type=float,
        default=0.8,
        help="Target resolution for the reconstruction.",
    )

    args = p.parse_args(argv)
    data_path = Path(args.data_path).resolve()
    config = Path(args.config).resolve()
    masks_folder = args.masks_path
    out_path = Path(args.out_path).resolve()
    target_res = args.target_res
    participant_label = args.participant_label
    use_svrtk_auto = args.use_svrtk_auto
    fake_run = args.fake_run
    if use_svrtk_auto:
        assert masks_folder is None, "Cannot use masks with svrtk:auto-2.20"
    else:
        masks_folder = Path(masks_folder).resolve()
    # Load a dictionary of subject-session-paths
    # sub_ses_dict = iter_dir(data_path, add_run_only=True)

    bids_layout = BIDSLayout(data_path, validate=True)

    with open(config, "r") as f:
        params = json.load(f)
    # Iterate over all subjects and sessions
    iterate = partial(
        iterate_subject,
        bids_layout=bids_layout,
        config_path=config,
        data_path=data_path,
        output_path=out_path,
        target_res=target_res,
        masks_folder=masks_folder,
        participant_label=participant_label,
        use_svrtk_auto=use_svrtk_auto,
        fake_run=fake_run,
    )
    for sub, config_sub in params.items():
        iterate(sub, config_sub)


if __name__ == "__main__":
    main()
