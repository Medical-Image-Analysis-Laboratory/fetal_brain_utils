import csv
import operator
from collections import defaultdict
from functools import reduce
import json
from pathlib import Path
import re
import copy
import numpy as np
import nibabel as ni
import os
from bids.layout.writing import build_path

AUTO_MASK_PATH = "/media/tsanchez/tsanchez_data/data/out_anon/masks"
OUT_JSON_ORDER = [
    "sr-id",
    "session",
    "ga",
    "stacks",
    "use_auto_mask",
    "config_path",
    "info",
    "im_path",
    "mask_path",
]


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


def find_run_id(file_list):
    run_dict = {
        int(re.findall(r"run-(\d+)_", str(file))[-1]): file
        for file in file_list
    }
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
    return [str(run_dict[s]) for s in stacks], auto_masks


def filter_run_list(stacks, run_list):
    run_dict = find_run_id(run_list)
    return [str(run_dict[s]) for s in stacks]


def get_cropped_stack_based_on_mask(
    image_ni, mask_ni, boundary_i=0, boundary_j=0, boundary_k=0, unit="mm"
):
    """
    Crops the input image to the field of view given by the bounding box
    around its mask.
    Original code by Michael Ebner:
    https://github.com/gift-surg/NiftyMIC/blob/master/niftymic/base/stack.py

    Input
    -----
    image_ni:
        Nifti image
    mask_ni:
        Corresponding nifti mask
    boundary_i:
    boundary_j:
    boundary_k:
    unit:
        The unit defining the dimension size in nifti

    Output
    ------
    image_cropped:
        Image cropped to the bounding box of mask_ni
    mask_cropped
        Mask cropped to its bounding box
    """

    image_ni = copy.deepcopy(image_ni)
    image = image_ni.get_fdata().squeeze()
    mask = mask_ni.get_fdata().squeeze()
    # Get rectangular region surrounding the masked voxels
    [x_range, y_range, z_range] = get_rectangular_masked_region(mask)

    if np.array([x_range, y_range, z_range]).all() is None:
        print("Cropping to bounding box of mask led to an empty image.")
        return None

    if unit == "mm":
        spacing = image_ni.header.get_zooms()
        boundary_i = np.round(boundary_i / float(spacing[0]))
        boundary_j = np.round(boundary_j / float(spacing[1]))
        boundary_k = np.round(boundary_k / float(spacing[2]))

    shape = image.shape
    x_range[0] = np.max([0, x_range[0] - boundary_i])
    x_range[1] = np.min([shape[0], x_range[1] + boundary_i])

    y_range[0] = np.max([0, y_range[0] - boundary_j])
    y_range[1] = np.min([shape[1], y_range[1] + boundary_j])

    z_range[0] = np.max([0, z_range[0] - boundary_k])
    z_range[1] = np.min([shape[2], z_range[1] + boundary_k])
    # Crop to image region defined by rectangular mask

    new_origin = list(
        ni.affines.apply_affine(
            mask_ni.affine, [x_range[0], y_range[0], z_range[0]]
        )
    ) + [1]
    new_affine = image_ni.affine
    new_affine[:, -1] = new_origin
    image_cropped = crop_image_to_region(image, x_range, y_range, z_range)
    image_cropped = ni.Nifti1Image(image_cropped, new_affine)
    return image_cropped


def crop_image_to_region(
    image: np.ndarray,
    range_x: np.ndarray,
    range_y: np.ndarray,
    range_z: np.ndarray,
) -> np.ndarray:
    """
    Crop given image to region defined by voxel space ranges
    Original code by Michael Ebner:
    https://github.com/gift-surg/NiftyMIC/blob/master/niftymic/base/stack.py

    Input
    ------
    image: np.array
        image which will be cropped
    range_x: (int, int)
        pair defining x interval in voxel space for image cropping
    range_y: (int, int)
        pair defining y interval in voxel space for image cropping
    range_z: (int, int)
        pair defining z interval in voxel space for image cropping

    Output
    ------
    image_cropped:
        The image cropped to the given x-y-z region.
    """
    image_cropped = image[
        range_x[0] : range_x[1],
        range_y[0] : range_y[1],
        range_z[0] : range_z[1],
    ]
    return image_cropped
    # Return rectangular region surrounding masked region.
    #  \param[in] mask_sitk sitk.Image representing the mask
    #  \return range_x pair defining x interval of mask in voxel space
    #  \return range_y pair defining y interval of mask in voxel space
    #  \return range_z pair defining z interval of mask in voxel space


def get_rectangular_masked_region(
    mask: np.ndarray,
) -> tuple:
    """
    Computes the bounding box around the given mask
    Original code by Michael Ebner:
    https://github.com/gift-surg/NiftyMIC/blob/master/niftymic/base/stack.py

    Input
    -----
    mask: np.ndarray
        Input mask
    range_x:
        pair defining x interval of mask in voxel space
    range_y:
        pair defining y interval of mask in voxel space
    range_z:
        pair defining z interval of mask in voxel space
    """
    if np.sum(abs(mask)) == 0:
        return None, None, None
    shape = mask.shape
    # Compute sum of pixels of each slice along specified directions
    sum_xy = np.sum(mask, axis=(0, 1))  # sum within x-y-plane
    sum_xz = np.sum(mask, axis=(0, 2))  # sum within x-z-plane
    sum_yz = np.sum(mask, axis=(1, 2))  # sum within y-z-plane

    # Find masked regions (non-zero sum!)
    range_x = np.zeros(2)
    range_y = np.zeros(2)
    range_z = np.zeros(2)

    # Non-zero elements of numpy array nda defining x_range
    ran = np.nonzero(sum_yz)[0]
    range_x[0] = np.max([0, ran[0]])
    range_x[1] = np.min([shape[0], ran[-1] + 1])

    # Non-zero elements of numpy array nda defining y_range
    ran = np.nonzero(sum_xz)[0]
    range_y[0] = np.max([0, ran[0]])
    range_y[1] = np.min([shape[1], ran[-1] + 1])

    # Non-zero elements of numpy array nda defining z_range
    ran = np.nonzero(sum_xy)[0]
    range_z[0] = np.max([0, ran[0]])
    range_z[1] = np.min([shape[2], ran[-1] + 1])

    # Numpy reads the array as z,y,x coordinates! So swap them accordingly
    return (
        range_x.astype(int),
        range_y.astype(int),
        range_z.astype(int),
    )


class nested_defaultdict:
    """Convenience class to create an arbitrary nested dictionary
    using defaultdict. The dictionary can be accessed using tuples
    of keys (k1,k2,k3,...).
    """

    def __init__(self):
        self._nested_dd = self.nested_default_dict()

    def __repr__(self):
        return json.dumps(self._nested_dd)

    def __str__(self):
        return json.dumps(self._nested_dd)

    def nested_default_dict(self):
        """Define a nested default dictionary"""
        return defaultdict(self.nested_default_dict)

    def get(self, map_list):
        """Get an item from a nested dictionary using a tuple of keys"""
        return reduce(operator.getitem, map_list, self._nested_dd)

    def set(self, map_list, value):
        """Set an item in a nested dictionary using a tuple of keys"""
        self.get(map_list[:-1])[map_list[-1]] = value

    def to_dict(self):
        return json.loads(json.dumps(self._nested_dd))


def csv_to_list(csv_path):
    file_list = []
    reader = csv.DictReader(open(csv_path))
    for i, line in enumerate(reader):
        file_list.append(line)
    return file_list


def iter_bids_dict(bids_dict: dict, _depth=0, max_depth=1):
    """Return a single iterator over the dictionary obtained from
    iter_dir - flexibly handles cases with and without a session date.
    Taken from https://thispointer.com/python-how-to-iterate-over-
    nested-dictionary-dict-of-dicts/
    """
    assert _depth >= 0
    for key, value in bids_dict.items():
        if isinstance(value, dict) and _depth < max_depth:
            # If value is dict then iterate over all its values
            for keyvalue in iter_bids_dict(
                value, _depth + 1, max_depth=max_depth
            ):
                yield (key, *keyvalue)
        else:
            # If value is not dict type then yield the value
            yield (key, value)


def iter_dir(
    dir: str,
    suffix: str = ".nii.gz",
    list_id: list = False,
    add_run_only: bool = False,
):
    """Iterate a BIDS-like directory with a structure
    subject-session-anat-scan and list all files in each
    branch with the given suffix.

    return_id:
        Whether the function should list the run-id instead
        of the file path.
    add_run_only:
        Whether only the files with run- should be added.
        Filtering out some additional files in the anat folder.
    """
    print("Checking ", dir)
    dir = Path(dir)
    subject_dict = dict()
    for subject in sorted(dir.iterdir()):
        if "sub-" not in subject.stem:
            continue
        subject_str = str(subject.stem).replace("sub-", "")
        session_dict = dict()
        for session in sorted(subject.iterdir()):
            if "ses-" not in session.stem:
                continue
            session_str = str(session.stem).replace("ses-", "")
            sub_ses_list = []
            for file in sorted((session / "anat").iterdir()):
                if suffix is not None:
                    if not file.name.endswith(suffix):
                        continue
                if str(file.name).startswith("."):
                    continue
                ind = re.findall(r"run-(\d+)_", str(file))
                if add_run_only and len(ind) < 1:
                    continue
                sub_ses_list += [ind[-1]] if list_id else [file]
            session_dict[session_str] = sub_ses_list  # config_path/json_file
        subject_dict[subject_str] = session_dict
    return subject_dict


def iter_bids(
    bids_layout,
    extension="nii.gz",
    datatype="anat",
    suffix="T2w",
    target=None,
    return_type="filename",
    skip_run=False,
    return_all=False,
):
    """Return a single iterator over the BIDSLayout obtained from
    pybids - flexibly handles cases with and without a session date.
    """
    for sub in sorted(bids_layout.get_subjects()):
        for ses in [None] + sorted(bids_layout.get_sessions(subject=sub)):
            if skip_run:
                run_list = [None]
            else:
                run_list = sorted(
                    bids_layout.get_runs(subject=sub, session=ses)
                )
            for run in run_list:
                out = bids_layout.get(
                    subject=sub,
                    session=ses,
                    run=run,
                    extension=extension,
                    datatype=datatype,
                    suffix=suffix,
                    target=target,
                    return_type=return_type,
                )
                if return_all:
                    for o in out:
                        yield (sub, ses, run, o)
                else:
                    if len(out) > 0:
                        yield (sub, ses, run, out[0])
