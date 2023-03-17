import pytest
from fetal_brain_utils.utils import get_mask_path, iter_dir, iter_bids
from bids import BIDSLayout
from pathlib import Path
import json

FILE_DIR = Path(__file__).parent.resolve()
BIDS_DIR = FILE_DIR / "data"
MASKS_DIR = FILE_DIR / "data/derivatives/masks"


@pytest.mark.parametrize(
    "bids_dir,sub,ses,run,out",
    [
        (
            "/test",
            "test01",
            "01",
            "1",
            "/test/sub-test01/ses-01/anat/sub-test01_ses-01_acq-haste_run-1_T2w_mask.nii.gz",
        ),
        (
            "/test",
            "test01",
            None,
            1,
            "/test/sub-test01/anat/sub-test01_acq-haste_run-1_T2w_mask.nii.gz",
        ),
    ],
)
def test_get_mask_path(bids_dir, sub, ses, run, out):
    assert get_mask_path(bids_dir, sub, ses, run) == out


def test_iter_bids():
    layout = BIDSLayout(BIDS_DIR)
    out = [list(o) for o in iter_bids(layout)]
    with open(FILE_DIR / "output/iter_bids_dir.json", "r") as f:
        ref = json.load(f)
    assert out == ref


def test_iter_bids_mask():
    layout = BIDSLayout(MASKS_DIR, validate=False)
    out = [list(o) for o in iter_bids(layout, suffix="mask")]
    with open(FILE_DIR / "output/iter_bids_dir_mask.json", "r") as f:
        ref = json.load(f)
    assert out == ref
