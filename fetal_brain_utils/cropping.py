import copy
import numpy as np
import nibabel as ni
from .definitions import AUTO_MASK_PATH
from .utils import squeeze_dim
from functools import partial
from pathlib import Path
import os


def crop_input(
    sub,
    ses,
    output_path,
    img_list,
    mask_list,
    mask_input,
    fake_run=False,
    boundary_ip=15,
    boundary_tp=0,
):

    sub_ses_output = output_path / f"sub-{sub}/ses-{ses}/anat"
    if not fake_run:
        os.makedirs(sub_ses_output, exist_ok=True)

    crop_path = partial(
        get_cropped_stack_based_on_mask,
        boundary_i=boundary_ip,
        boundary_j=boundary_ip,
        boundary_k=boundary_tp,
    )
    im_list_c, mask_list_c = [], []
    for image, mask in zip(img_list, mask_list):
        print(f"Processing {image} {mask}")
        im_file, mask_file = Path(image).name, Path(mask).name
        cropped_im_path = sub_ses_output / im_file
        cropped_mask_path = sub_ses_output / mask_file
        im, m = ni.load(image), ni.load(mask)

        imc = crop_path(im, m)
        maskc = crop_path(m, m)
        # Masking

        if imc is not None:
            if mask_input:
                imc = ni.Nifti1Image(imc.get_fdata() * maskc.get_fdata(), imc.affine)
            else:
                imc = ni.Nifti1Image(imc.get_fdata(), imc.affine)
            if not fake_run:
                ni.save(imc, cropped_im_path)
                ni.save(maskc, cropped_mask_path)
            im_list_c.append(str(cropped_im_path))
            mask_list_c.append(str(cropped_mask_path))
    return im_list_c, mask_list_c


def get_cropped_stack_based_on_mask(
    image_ni, mask_ni, boundary_i=0, boundary_j=0, boundary_k=0, unit="mm"
):
    """
    Crops the input image to the field of view given by the bounding box
    around its mask.
    Code inspired from Michael Ebner: https://github.com/gift-surg/NiftyMIC/blob/master/niftymic/base/stack.py

    Input
    -----
    image_ni: Nifti image
        Nifti image
    mask_ni: Nifti image
        Corresponding nifti mask
    boundary_i: int
        Boundary to add to the bounding box in the i direction
    boundary_j: int
        Boundary to add to the bounding box in the j direction
    boundary_k: int
        Boundary to add to the bounding box in the k direction
    unit: str
        The unit defining the dimension size in nifti

    Output
    ------
    image_cropped:
        Image cropped to the bounding box of mask_ni, including boundary
    mask_cropped
        Mask cropped to its bounding box
    """

    image_ni = copy.deepcopy(image_ni)

    image = squeeze_dim(image_ni.get_fdata(), -1)
    mask = squeeze_dim(mask_ni.get_fdata(), -1)

    assert all(
        [i >= m] for i, m in zip(image.shape, mask.shape)
    ), "For a correct cropping, the image should be larger or equal to the mask."

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

    shape = [min(im, m) for im, m in zip(image.shape, mask.shape)]
    x_range[0] = np.max([0, x_range[0] - boundary_i])
    x_range[1] = np.min([shape[0], x_range[1] + boundary_i])

    y_range[0] = np.max([0, y_range[0] - boundary_j])
    y_range[1] = np.min([shape[1], y_range[1] + boundary_j])

    z_range[0] = np.max([0, z_range[0] - boundary_k])
    z_range[1] = np.min([shape[2], z_range[1] + boundary_k])

    new_origin = list(
        ni.affines.apply_affine(mask_ni.affine, [x_range[0], y_range[0], z_range[0]])
    ) + [1]

    new_affine = image_ni.affine
    new_affine[:, -1] = new_origin

    image_cropped = image[
        x_range[0] : x_range[1],
        y_range[0] : y_range[1],
        z_range[0] : z_range[1],
    ]

    image_cropped = ni.Nifti1Image(image_cropped, new_affine)
    return image_cropped


def get_rectangular_masked_region(
    mask: np.ndarray,
) -> tuple:
    """
    Computes the bounding box around the given mask
    Code inspired from Michael Ebner: https://github.com/gift-surg/NiftyMIC/blob/master/niftymic/base/stack.py

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
    # Define the dimensions along which to sum the data
    sum_axis = [(1, 2), (0, 2), (0, 1)]
    range_list = []

    # Non-zero elements of numpy array along the the 3 dimensions
    for i in range(3):
        sum_mask = np.sum(mask, axis=sum_axis[i])
        ran = np.nonzero(sum_mask)[0]

        low = np.max([0, ran[0]])
        high = np.min([shape[i], ran[-1] + 1])
        range_list.append(np.array([low, high]).astype(int))

    return range_list
