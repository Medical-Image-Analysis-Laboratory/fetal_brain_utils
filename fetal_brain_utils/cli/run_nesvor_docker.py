"""
Run the NeSVor pipeline on data.
"""

from fetal_brain_utils import iter_dir, filter_run_list, find_run_id, crop_input
from fetal_brain_utils.definitions import OUT_JSON_ORDER
from bids.layout.writing import build_path
import argparse
from pathlib import Path
import os
from functools import partial
import json
import re
import nibabel as ni
import pdb
from bids.layout import BIDSLayout
import shutil

# Default data path
DATA_PATH = Path("/media/tsanchez/tsanchez_data/data/data")
AUTO_MASK_PATH = "/media/tsanchez/tsanchez_data/data/out_anon/masks"

BATCH_SIZE = 4096
NESVOR_VERSION = "v0.5.0"


def iterate_subject(
    sub,
    config_sub,
    bids_layout,
    data_path,
    output_path,
    mask_base_path,
    participant_label,
    target_res,
    config,
    nesvor_version,
    batch_size,
    recon_type,
    mask_input,
    fake_run,
    bias_field_correction,
    extra_commands,
):
    if participant_label:
        if sub not in participant_label:
            return
    if sub not in bids_layout.get_subjects():
        print(f"Subject {sub} not found in {data_path}")
        return

    if recon_type == "svr":
        assert len(target_res) == 1, "Only one resolution can be used for SVR."
        # if sub not in sub_ses_dict:
        #    print(f"Subject {sub} not found in {data_path}")
        #    return
        output_path = Path(output_path)

    mask_on = mask_base_path is not None
    if mask_on:
        mask_base_path = Path(mask_base_path)
        masks_layout = BIDSLayout(mask_base_path, validate=False)
        output_path_crop = output_path / "cropped_input"
    output_path = output_path / "nesvor"
    if not fake_run:
        os.makedirs(output_path, exist_ok=True)

    sub_path = f"sub-{sub}"
    if not isinstance(config_sub, list):
        config_sub = [config_sub]

    for conf in config_sub:
        ses = conf["session"] if "session" in conf else None
        img_list = bids_layout.get_runs(
            subject=sub,
            session=ses,
            extension="nii.gz",
            return_type="filename",
        )
        if mask_on:
            mask_list = masks_layout.get_runs(
                subject=sub,
                session=ses,
                extension="nii.gz",
                return_type="filename",
            )
        stacks = conf["stacks"] if "stacks" in conf else find_run_id(img_list)
        run_id = conf["sr-id"] if "sr-id" in conf else "1"
        run_path = f"run-{run_id}"

        img_list = [str(f) for f in filter_run_list(stacks, img_list)]
        conf["im_path"] = img_list

        if mask_on:
            mask_list = [str(f) for f in mask_list]
            mask_list = [str(f) for f in filter_run_list(stacks, mask_list)]
            conf["mask_path"] = mask_list

        conf["config_path"] = str(config)
        ses_path = f"ses-{ses}"
        # Construct the data and mask path from their respective
        # base paths
        output_sub_ses = (
            output_path / sub_path / ses_path / "anat" if ses else output_path / sub_path / "anat"
        )
        if not fake_run:
            os.makedirs(output_sub_ses, exist_ok=True)
        if mask_on:
            img_list, mask_list = crop_input(
                sub,
                ses,
                output_path_crop,
                img_list,
                mask_list,
                mask_input,
                fake_run,
            )

        mount_base = Path(img_list[0]).parent
        img_str = " ".join([str(Path("/data") / Path(im).name) for im in img_list])
        if mask_on:
            mask_str = " ".join([str(Path("/data") / Path(m).name) for m in mask_list])

        out = Path("/out")
        model = out / f"{sub_path}_{ses_path}_{run_path}_model.pt"

        for i, res in enumerate(target_res):
            res_str = str(res).replace(".", "p")
            out_base = (
                f"{sub_path}_{ses_path}_" f"res-{res_str}_{run_path}_T2w"
                if ses
                else f"{sub_path}_res-{res_str}_{run_path}_rec-nesvor_T2w"
            )
            if nesvor_version == "v0.5.0":
                output_file = str(out / out_base) + ".nii.gz"
            else:
                output_file = str(out / out_base) + "_misaligned.nii.gz"
            output_json = str(output_sub_ses / out_base) + ".json"
            if i == 0:
                nesvor_arg = "svr" if recon_type == "svr" else "reconstruct"
                cmd = (
                    f"docker run --gpus '\"device=0\"' "
                    f"-v {mount_base}:/data "
                    f"-v {output_sub_ses}:/out "
                    f"junshenxu/nesvor:{nesvor_version} nesvor {nesvor_arg} "
                    f"--input-stacks {img_str} "
                    f"--output-volume {output_file} "
                )
                if mask_on:
                    cmd += f"--stack-masks {mask_str} "
                if recon_type == "nesvor":
                    cmd += (
                        f"--output-resolution {res} "
                        f"--output-model {model} "
                        f"--batch-size {batch_size} "
                        " --n-proc-n4 1 "
                    )
                if nesvor_version == "v0.5.0" and bias_field_correction:
                    cmd += "--bias-field-correction"
                    cmd += "--n-levels-bias 1 "
                if extra_commands != "":
                    cmd += f" {extra_commands}"
            else:
                cmd = (
                    f"docker run --gpus '\"device=0\"' "
                    f"-v {output_sub_ses}:/out "
                    f"junshenxu/nesvor:{nesvor_version} nesvor sample-volume "
                    f"--input-model {model} "
                    f"--output-resolution {res} "
                    f"--output-volume {output_file} "
                    f"--output-resolution {res} "
                    f"--inference-batch-size {batch_size*2}"
                )
            conf["info"] = {
                "reconstruction": "NeSVoR",
                "res": res,
                "model": str(model),
                "command": cmd,
            }
            print(cmd)
            if not fake_run:
                conf = {k: conf[k] for k in OUT_JSON_ORDER if k in conf.keys()}
                with open(output_json, "w") as f:
                    json.dump(conf, f, indent=4)

                os.system(cmd)

                # Transform the affine of the sr reconstruction
                if nesvor_version != "v0.5.0":
                    out_file = str(output_sub_ses / out_base) + "_misaligned.nii.gz"
                    out_file_reo = str(output_sub_ses / out_base) + ".nii.gz"
                    sr = ni.load(out_file)
                    affine = sr.affine[[2, 1, 0, 3]]
                    affine[1, :] *= -1
                    ni.save(
                        ni.Nifti1Image(sr.get_fdata()[:, :, :], affine, sr.header),
                        out_file_reo,
                    )


def main(argv=None):
    from .parser import get_default_parser

    p = get_default_parser("NeSVoR (docker)")

    p.add_argument(
        "--target_res",
        required=True,
        nargs="+",
        type=float,
        help="Target resolutions at which the reconstruction should be done.",
    )

    p.add_argument(
        "--version",
        default=NESVOR_VERSION,
        type=str,
        choices=["v0.1.0", "v0.5.0"],
        help="Version of NeSVoR to use.",
    )

    p.add_argument(
        "--mask_input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the input stacks should be masked prior to computation.",
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for the NeSVoR pipeline.",
    )

    p.add_argument(
        "--recon_type",
        type=str,
        default="nesvor",
        choices=["nesvor", "svr"],
        help="Types of reconstruction to be run: train a NeSVoR model or just run SVR.",
    )

    p.add_argument(
        "--extra_commands",
        type=str,
        default="",
        help="Extra commands to be added to the NeSVoR command.",
    )

    p.add_argument(
        "--bias_field_correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether the input stacks should be bias corrected prior to  computation.",
    )
    args = p.parse_args(argv)

    data_path = Path(args.data_path).resolve()
    out_path = Path(args.out_path).resolve()
    masks_on = args.masks_path is not None
    if masks_on:
        masks_folder = Path(args.masks_path).resolve()
    else:
        masks_folder = None
    config = Path(args.config).resolve()

    bids_layout = BIDSLayout(data_path, validate=False)

    with open(config, "r") as f:
        params = json.load(f)
    # Iterate over all subjects and sessions
    iterate = partial(
        iterate_subject,
        bids_layout=bids_layout,
        data_path=data_path,
        output_path=out_path,
        mask_base_path=masks_folder,
        participant_label=args.participant_label,
        target_res=args.target_res,
        config=config,
        nesvor_version=args.version,
        batch_size=args.batch_size,
        mask_input=args.mask_input,
        recon_type=args.recon_type,
        extra_commands=args.extra_commands,
        bias_field_correction=args.bias_field_correction,
        fake_run=args.fake_run,
    )
    for sub, config_sub in params.items():
        iterate(sub, config_sub)


if __name__ == "__main__":
    main()
