"""
Run the NeSVor pipeline on data.
"""
from fetal_brain_utils import iter_dir
from fetal_brain_utils.definitions import OUT_JSON_ORDER
from bids.layout.writing import build_path
import argparse
from pathlib import Path
import os
from functools import partial
import json
import re
import nibabel as ni

# Default data path
DATA_PATH = Path("/media/tsanchez/tsanchez_data/data/data")
AUTO_MASK_PATH = "/media/tsanchez/tsanchez_data/data/out_anon/masks"

BATCH_SIZE = 8192
NESVOR_VERSION = "v0.5.0"


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
    run_dict = {int(re.findall(r"run-(\d+)_", str(file))[-1]): file for file in file_list}
    return run_dict


def filter_run_list(stacks, run_list):
    run_dict = find_run_id(run_list)
    return [run_dict[s] for s in stacks]


def crop_input(sub, ses, output_path, img_list, mask_list, mask_input, fake_run=False):
    import nibabel as ni
    from fetal_brain_utils import get_cropped_stack_based_on_mask
    from functools import partial

    sub_ses_output = output_path / f"sub-{sub}/ses-{ses}/anat"
    if not fake_run:
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
        if not fake_run:
            im, m = ni.load(image), ni.load(mask)

            imc = crop_path(im, m)
            maskc = crop_path(m, m)
            # Masking
            if mask_input:
                imc = ni.Nifti1Image(imc.get_fdata() * maskc.get_fdata(), imc.affine)
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
    sub_ses_dict,
    data_path,
    output_path,
    mask_base_path,
    participant_label,
    target_res,
    config,
    nesvor_version,
    mask_input,
    fake_run,
):
    if participant_label:
        if sub not in participant_label:
            return
    if sub not in sub_ses_dict:
        print(f"Subject {sub} not found in {data_path}")
        return

    mask_base_path = Path(mask_base_path)
    output_path = Path(output_path)
    output_path_crop = output_path / "cropped_input"
    output_path = output_path / "nesvor"
    sub_ses_masks_dict = iter_dir(mask_base_path)
    if not fake_run:
        os.makedirs(output_path, exist_ok=True)

    sub_path = f"sub-{sub}"
    if not isinstance(config_sub, list):
        config_sub = [config_sub]

    for conf in config_sub:
        if "session" not in conf:
            ses = None
            mask_list = sub_ses_masks_dict[sub]
            img_list = sub_ses_dict[sub]
        else:
            ses = conf["session"]
            mask_list = sub_ses_masks_dict[sub][ses]
            img_list = sub_ses_dict[sub][ses]
        stacks = conf["stacks"] if "stacks" in conf else find_run_id(img_list)
        run_id = conf["sr-id"] if "sr-id" in conf else "1"
        run_path = f"run-{run_id}"

        mask_list = [str(f) for f in mask_list]
        img_list = [str(f) for f in filter_run_list(stacks, img_list)]
        mask_list = [str(f) for f in filter_run_list(stacks, mask_list)]
        conf["im_path"] = img_list
        conf["mask_path"] = mask_list
        conf["config_path"] = str(config)
        ses_path = f"ses-{ses}"
        # Construct the data and mask path from their respective
        # base paths
        output_sub_ses = output_path / sub_path / ses_path / "anat"
        if not fake_run:
            os.makedirs(output_sub_ses, exist_ok=True)

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
        mask_str = " ".join([str(Path("/data") / Path(m).name) for m in mask_list])

        out = Path("/out")
        model = out / f"{sub_path}_{ses_path}_{run_path}_model.pt"

        for i, res in enumerate(target_res):
            res_str = str(res).replace(".", "p")
            out_base = f"{sub_path}_{ses_path}_" f"acq-haste_res-{res_str}_{run_path}_T2w"
            if nesvor_version == "v0.5.0":
                output_file = str(out / out_base) + ".nii.gz"
            else:
                output_file = str(out / out_base) + "_misaligned.nii.gz"
            output_json = str(output_sub_ses / out_base) + ".json"
            if i == 0:
                cmd = (
                    f"docker run --gpus '\"device=0\"' "
                    f"-v {mount_base}:/data "
                    f"-v {output_sub_ses}:/out "
                    f"junshenxu/nesvor:{nesvor_version} nesvor reconstruct "
                    f"--input-stacks {img_str} "
                    f"--stack-masks {mask_str} "
                    f"--output-volume {output_file} "
                    f"--output-resolution {res} "
                    f"--output-model {model} "
                    f"--n-levels-bias 1 "
                    f"--batch-size {BATCH_SIZE} "
                )
                if nesvor_version == "v0.5.0":
                    cmd += "--bias-field-correction"
            else:
                cmd = (
                    f"docker run --gpus '\"device=0\"' "
                    f"-v {output_sub_ses}:/out "
                    f"junshenxu/nesvor:v0.1.0 nesvor sample-volume "
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
    args = p.parse_args(argv)

    data_path = Path(args.data_path).resolve()
    out_path = Path(args.out_path).resolve()
    masks_folder = Path(args.masks_path).resolve()
    config = Path(args.config).resolve()

    # Load a dictionary of subject-session-paths
    sub_ses_dict = iter_dir(data_path, add_run_only=True)

    with open(config, "r") as f:
        params = json.load(f)
    # Iterate over all subjects and sessions
    iterate = partial(
        iterate_subject,
        sub_ses_dict=sub_ses_dict,
        data_path=data_path,
        output_path=out_path,
        mask_base_path=masks_folder,
        participant_label=args.participant_label,
        target_res=args.target_res,
        config=config,
        nesvor_version=args.version,
        mask_input=args.mask_input,
        fake_run=args.fake_run,
    )
    for sub, config_sub in params.items():
        iterate(sub, config_sub)


if __name__ == "__main__":
    main()
