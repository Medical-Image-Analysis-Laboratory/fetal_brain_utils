"""
This module contains functions to convert BIDS directories to nnUNet format.
"""
import argparse
from bids import BIDSLayout
import os
import shutil
import numpy as np
import random
import SimpleITK as sitk
import json
import re

# Create a main function with a parser that parses a bids_directory and create a bids_layout

DATASET_JSON = {
    "channel_names": {"0": "T2"},
    "labels": {
        "background": 0,
        "External Cerebrospinal Fluid": 1,
        "Grey Matter": 2,
        "White Matter": 3,
        "Ventricles": 4,
        "Cerebellum": 5,
        "Deep Grey Matter": 6,
        "Brainstem": 7,
    },
    "numTraining": None,
    "file_ending": ".nii.gz",
}


def bids_to_nnunet_train(bids_layout, dataset_name, dataset_num, num_aug):
    """
    Create a dataset for nnUNet training from a BIDSLayout.
    The dataset will be created in the nnUNet_raw folder, and will be named
    Dataset<dataset_num>_<dataset_name>. The dataset will be created with the
    following structure:
    Dataset<dataset_num>_<dataset_name>
    ├── dataset.json
    ├── imagesTr
    │   ├── <bids_directory_folder_name>_<sub><run>_0000.nii.gz
    │   ├── <bids_directory_folder_name>_<sub><run>_0001.nii.gz
    │   ├── ...
    │   └── <bids_directory_folder_name>_<sub><run>_<num_aug-1>.nii.gz
    └── labelsTr

    Parameters
    ----------
    bids_layout : BIDSLayout
        BIDSLayout of the BIDS directory.
    dataset_name : str
        Name of the dataset.
    dataset_num : int
        Number of the dataset (nnUNet convention).
    num_aug : int
        Number of augmentations to create for each subject.

    """
    rec_seg = []
    for sub in sorted(bids_layout.get_subjects()):
        info = [sub]
        for suffix in ["T2w", "dseg"]:
            out = bids_layout.get(
                subject=sub,
                extension="nii.gz",
                datatype="anat",
                suffix=suffix,
                target=None,
                return_type="filename",
            )
            info.append(out[0])
        rec_seg.append(info)

    # get the content of the nnUNet_raw environment variable
    nnunet_raw = os.environ.get("nnUNet_raw")
    dataset_str = f"Dataset{dataset_num:03d}_{dataset_name}"

    # Assert that the path to nnunet_raw / dataset_num does not exist
    dataset_path = os.path.join(nnunet_raw, dataset_str)
    assert not os.path.exists(dataset_path), f"{dataset_str} already exists in {nnunet_raw}"

    # Create the directory dataset_path
    os.makedirs(dataset_path)
    tr_path = os.path.join(dataset_path, "imagesTr")
    ltr_path = os.path.join(dataset_path, "labelsTr")
    os.makedirs(tr_path)
    os.makedirs(ltr_path)

    # Create the dataset.json file with the template above
    dataset_json = DATASET_JSON
    dataset_json["numTraining"] = len(rec_seg) * num_aug
    # Save the dataset_json file in the dataset_path

    with open(os.path.join(dataset_path, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)

    # Iterate through the rec_seg list and copy the files to the dataset_path
    # and rename them to the format <bids_directory_folder_name>_<sub><run>_0000.nii.gz
    for sub, t2, dseg in rec_seg:
        for j in range(num_aug):
            t2_new_name = f"{dataset_name}_{sub}{j}_0000.nii.gz"
            dseg_new_name = f"{dataset_name}_{sub}{j}.nii.gz"
            t2_img = sitk.ReadImage(t2)
            dseg_img = sitk.ReadImage(dseg)

            t2_data = sitk.GetArrayFromImage(t2_img)
            dseg_data = sitk.GetArrayFromImage(dseg_img)
            mask = dseg_data > 0
            t2_data = t2_data * mask
            axes = list(range(t2_data.ndim))
            random.shuffle(axes)

            t2_data = np.transpose(t2_data, axes)
            dseg_data = np.transpose(dseg_data, axes)
            # Flip the axes randomly
            for axis in axes:
                if random.random() < 0.5:
                    t2_data = np.flip(t2_data, axis=axis)
                    dseg_data = np.flip(dseg_data, axis=axis)
            # Save the new images
            t2_new = sitk.GetImageFromArray(t2_data)
            dseg_new = sitk.GetImageFromArray(dseg_data)
            t2_new.CopyInformation(t2_img)
            dseg_new.CopyInformation(dseg_img)
            sitk.WriteImage(t2_new, os.path.join(tr_path, t2_new_name))
            sitk.WriteImage(dseg_new, os.path.join(ltr_path, dseg_new_name))


def bids_to_nnunet_test(bids_layout, dataset_num):
    """Format the data from a BIDS directory to nnUNet format for testing
    Note: Currently, this does *not* handle reconstructed images as it requires a run number.

    Parameters
    ----------
    bids_layout : BIDSLayout
        BIDSLayout of the BIDS directory.
    dataset_num : int
        Number of the dataset (nnUNet convention).

    """
    rec_seg = []
    for sub in sorted(bids_layout.get_subjects()):
        out = bids_layout.get(
            subject=sub,
            extension="nii.gz",
            datatype="anat",
            suffix="T2w",
            target=None,
            return_type="filename",
        )
        for st in out:
            rec_seg.append([sub, int(re.findall(r"run-(\d+)_", str(st))[-1]), st])

    # get the content of the nnUNet_raw environment variable
    nnunet_raw = os.environ.get("nnUNet_raw")
    dataset_str = f"Dataset{dataset_num:03d}_"
    # Check whether a folder containing the dataset_str exists at nnunet_raw
    dataset_name = "data"
    for s in os.listdir(nnunet_raw):
        if dataset_str in s:
            dataset_name = s.split("_")[1]
            break
    dataset_str = f"Dataset{dataset_num:03d}_{dataset_name}"
    # Assert that the path to nnunet_raw / dataset_num does not exist
    dataset_path = os.path.join(nnunet_raw, dataset_str)

    # Create the directory dataset_path
    ts_path = os.path.join(dataset_path, "imagesTs")
    assert not os.path.exists(ts_path), f"imagesTs already exists in {ts_path}"
    os.makedirs(ts_path)

    for sub, run, t2 in rec_seg:
        t2_new_name = f"{dataset_name}_{sub}{run:02d}_0000.nii.gz"
        shutil.copy(t2, os.path.join(ts_path, t2_new_name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bids_directory", help="Path to the BIDS directory")
    parser.add_argument(
        "--dataset_num", default=1, type=int, help="Number of the dataset in the nnUNet folder"
    )
    parser.add_argument("--num_aug", default=3, type=int, help="Factor of data augmentation")
    parser.add_argument(
        "--test_data", action="store_true", default=False, help="Whether data are used for testing."
    )

    args = parser.parse_args()

    bids_layout = BIDSLayout(args.bids_directory, validate=False)
    print(bids_layout)
    dataset_name = args.bids_directory.split("/")
    dataset_name = dataset_name[-1] if dataset_name[-1] != "" else dataset_name[-2]
    dataset_name = dataset_name.replace("_", "")
    if not args.test_data:
        dataset_name = args.bids_directory.split("/")
        dataset_name = dataset_name[-1] if dataset_name[-1] != "" else dataset_name[-2]
        dataset_name = dataset_name.replace("_", "")
        bids_to_nnunet_train(bids_layout, dataset_name, args.dataset_num, args.num_aug)
    else:
        bids_to_nnunet_test(bids_layout, args.dataset_num)


if __name__ == "__main__":
    main()
