import copy
import numpy as np
import nibabel as ni
from .definitions import AUTO_MASK_PATH
from .utils import squeeze_dim
from functools import partial
from pathlib import Path
import os
from typing import List, Optional, Any, Dict

N4_PARAMS = {
    "n_proc_n4": 1,
    "shrink_factor_n4": 2,
    "tol_n4": 0.001,
    "spline_order_n4": 3,
    "noise_n4": 0.01,
    "n_iter_n4": 50,
    "n_levels_n4": 4,
    "n_control_points_n4": 4,
    "n_bins_n4": 200,
}


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
    denoise=False,
    correct_bias=False,
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
        print(f"Processing {image} {mask} -- denoise={denoise}")
        im_file, mask_file = Path(image).name, Path(mask).name
        cropped_im_path = sub_ses_output / im_file
        cropped_mask_path = sub_ses_output / mask_file
        im, m = ni.load(image), ni.load(mask)

        imc = crop_path(im, m)
        maskc = crop_path(m, m)
        # Masking
        if imc is not None:
            error = False
            if mask_input:
                imc = ni.Nifti1Image(imc.get_fdata() * maskc.get_fdata(), imc.affine, imc.header)
            else:
                imc = ni.Nifti1Image(imc.get_fdata(), imc.affine, imc.header)
            if not fake_run:
                ni.save(imc, cropped_im_path)
                ni.save(maskc, cropped_mask_path)
                if denoise:
                    os.system(
                        f"DenoiseImage -i {cropped_im_path} -n Gaussian -o {cropped_im_path} -s 1"
                    )

                if correct_bias:
                    res_x, res_y, res_z = imc.header.get_zooms()
                    imc = ni.load(cropped_im_path)
                    try:
                        im_corr = n4_bias_field_correction_single(
                            imc.get_fdata().astype(np.float32),
                            maskc.get_fdata().astype(np.uint8),
                            float(res_x),
                            float(res_y),
                            float(res_z),
                            n4_params=N4_PARAMS,
                        )
                        imc = ni.Nifti1Image(im_corr, imc.affine, imc.header)
                        ni.save(imc, cropped_im_path)
                    except Exception as e:
                        print(f"Error in bias correction -- Skipping the stack: {e}")
                        error = True
                        im_res = imc.header["pixdim"][1:4]

            im_res = imc.header["pixdim"][1:4]
            mask_res = maskc.header["pixdim"][1:4]
            im_aff = imc.affine
            mask_aff = maskc.affine
            im_shape = imc.shape
            mask_shape = maskc.shape
            if compare_resolution_affine(im_res, im_aff, mask_res, mask_aff, im_shape, mask_shape):
                if not error:
                    im_list_c.append(str(cropped_im_path))
                    mask_list_c.append(str(cropped_mask_path))
            else:
                print(f"Resolution/shape/affine mismatch -- Skipping the stack: {e}")

    return im_list_c, mask_list_c


def compare_resolution_affine(r1, a1, r2, a2, s1, s2) -> bool:
    r1 = np.array(r1)
    a1 = np.array(a1)
    r2 = np.array(r2)
    a2 = np.array(a2)
    if s1 != s2:
        return False
    if r1.shape != r2.shape:
        return False
    if np.amax(np.abs(r1 - r2)) > 1e-3:
        return False
    if a1.shape != a2.shape:
        return False
    if np.amax(np.abs(a1 - a2)) > 1e-3:
        return False
    return True


def n4_bias_field_correction_single(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    res_x: float,
    res_y: float,
    res_z: float,
    n4_params: Dict[str, Any],
) -> np.ndarray:
    import SimpleITK as sitk

    sitk_img_full = sitk.GetImageFromArray(image)
    sitk_img_full.SetSpacing([res_z, res_y, res_x])
    if mask is not None:
        sitk_mask_full = sitk.GetImageFromArray(mask)
        sitk_mask_full.SetSpacing([res_z, res_y, res_x])

    shrinkFactor = n4_params.get("shrink_factor_n4", 1)
    if shrinkFactor > 1:
        sitk_img = sitk.Shrink(sitk_img_full, [shrinkFactor] * sitk_img_full.GetDimension())
        sitk_mask = sitk.Shrink(sitk_mask_full, [shrinkFactor] * sitk_img_full.GetDimension())
    else:
        sitk_img = sitk_img_full
        sitk_mask = sitk_mask_full

    bias_field_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # see https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1N4BiasFieldCorrectionImageFilter.html for details
    if "fwhm_n4" in n4_params:  # bias_field_fwhm
        bias_field_corrector.SetBiasFieldFullWidthAtHalfMaximum(n4_params["fwhm_n4"])
    if "tol_n4" in n4_params:  # convergence_threshold
        bias_field_corrector.SetConvergenceThreshold(n4_params["tol_n4"])
    if "spline_order_n4" in n4_params:  # spline_order
        bias_field_corrector.SetSplineOrder(n4_params["spline_order_n4"])
    if "noise_n4" in n4_params:  # wiener_filter_noise
        bias_field_corrector.SetWienerFilterNoise(n4_params["noise_n4"])
    if "n_iter_n4" in n4_params and "n_levels_n4" in n4_params:
        # number_of_iteration, number_fitting_levels
        bias_field_corrector.SetMaximumNumberOfIterations(
            [n4_params["n_iter_n4"]] * n4_params["n_levels_n4"]
        )
    if "n_control_points_n4" in n4_params:  # number of control points
        bias_field_corrector.SetNumberOfControlPoints(n4_params["n_control_points_n4"])
    if "n_bins_n4" in n4_params:  # number of histogram bins
        bias_field_corrector.SetNumberOfHistogramBins(n4_params["n_bins_n4"])

    if mask is not None:
        corrected_sitk_img = bias_field_corrector.Execute(sitk_img, sitk_mask)
    else:
        corrected_sitk_img = bias_field_corrector.Execute(sitk_img)

    if shrinkFactor > 1:
        log_bias_field_full = bias_field_corrector.GetLogBiasFieldAsImage(sitk_img_full)
        corrected_sitk_img_full = sitk_img_full / sitk.Exp(log_bias_field_full)
    else:
        corrected_sitk_img_full = corrected_sitk_img

    corrected_image = sitk.GetArrayFromImage(corrected_sitk_img_full)

    return corrected_image


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
    affine = copy.deepcopy(image_ni.affine)
    header = copy.deepcopy(image_ni.header)

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

    new_origin = list(ni.affines.apply_affine(affine, [x_range[0], y_range[0], z_range[0]])) + [1]

    new_affine = affine
    new_affine[:, -1] = new_origin

    image_cropped = image[
        x_range[0] : x_range[1],
        y_range[0] : y_range[1],
        z_range[0] : z_range[1],
    ]
    image_cropped = ni.Nifti1Image(image_cropped, new_affine, header)
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
