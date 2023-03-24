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


def crop_input(sub, ses, output_path, img_list, mask_list, fake_run=False):
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

            imc = ni.Nifti1Image(imc.get_fdata() * maskc.get_fdata(), imc.affine)

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

        mask_list, auto_masks = filter_and_complement_mask_list(stacks, sub, ses, mask_list)
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
        if not fake_run:
            os.makedirs(output_sub_ses, exist_ok=True)

        img_list, mask_list = crop_input(sub, ses, output_path_crop, img_list, mask_list, fake_run)
        mount_base = Path(img_list[0]).parent
        img_str = " ".join([str(Path("/data") / Path(im).name) for im in img_list])
        mask_str = " ".join([str(Path("/data") / Path(m).name) for m in mask_list])

        out = Path("/out")
        model = out / f"{sub_path}_{ses_path}_{run_path}_model.pt"

        for i, res in enumerate(target_res):
            res_str = str(res).replace(".", "p")
            out_base = f"{sub_path}_{ses_path}_" f"acq-haste_res-{res_str}_{run_path}_T2w"
            output_file = str(out / out_base) + "_misaligned.nii.gz"
            output_json = str(output_sub_ses / out_base) + ".json"
            if i == 0:
                cmd = (
                    f"docker run --gpus '\"device=0\"' "
                    f"-v {mount_base}:/data "
                    f"-v {output_sub_ses}:/out "
                    f"junshenxu/nesvor:v0.1.0 nesvor reconstruct "
                    f"--input-stacks {img_str} "
                    f"--stack-masks {mask_str} "
                    f"--n-levels-bias 1 "
                    f"--output-volume {output_file} "
                    f"--output-resolution {res} "
                    f"--output-model {model} "
                    f"--batch-size {BATCH_SIZE}"
                )
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

    p = get_default_parser()

    p.add_argument(
        "--target_res",
        required=True,
        nargs="+",
        type=float,
        help="Target resolutions at which the reconstruction should be done.",
    )

    args = p.parse_args(argv)

    data_path = Path(args.data_path).resolve()
    out_path = Path(args.out_path).resolve()
    masks_folder = Path(args.masks_path).resolve()
    config = args.config

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
        mask_base_path=masks_folder,
        participant_label=args.participant_label,
        target_res=args.target_res,
        config=config,
        fake_run=args.fake_run,
    )
    for sub, config_sub in params.items():
        iterate(sub, config_sub)


if __name__ == "__main__":
    main()
