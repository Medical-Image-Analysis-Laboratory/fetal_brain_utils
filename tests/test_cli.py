from fetal_brain_utils.cli.run_niftymic import (
    iterate_subject as run_niftymic,
)
from fetal_brain_utils.cli.run_svrtk import (
    iterate_subject as run_svrtk,
)
import json
from bids import BIDSLayout
from pathlib import Path
import io
from contextlib import redirect_stdout


FILE_DIR = Path(__file__).parent.resolve()
BIDS_DIR = FILE_DIR / "data"
CONFIG_PATH = FILE_DIR / "data/code/params.json"
MASKS_DIR = FILE_DIR / "data/derivatives/masks"


def test_niftymic_iterate_subject():
    """Test the text output of run_niftymic."""
    with open(CONFIG_PATH, "r") as f:
        params = json.load(f)
    sub = "simu005"
    config_sub = params[sub]
    bids_layout = BIDSLayout(BIDS_DIR, validate=True)
    masks_folder = MASKS_DIR
    output_path = "out"

    f = io.StringIO()
    with redirect_stdout(f):
        run_niftymic(
            sub,
            config_sub,
            CONFIG_PATH,
            bids_layout,
            BIDS_DIR,
            output_path,
            masks_folder,
            alpha=0.068,
            participant_label=None,
            use_preprocessed=False,
            fake_run=True,
        )
    out = f.getvalue()

    with open(FILE_DIR / "output/niftymic_out.txt") as f:
        niftymic = f.readlines()
    niftymic = "".join(niftymic)

    assert out == niftymic


def test_svrtk_iterate_subject():
    """Test the text output of run_svrtk."""
    with open(CONFIG_PATH, "r") as f:
        params = json.load(f)
    sub = "simu005"
    config_sub = params[sub]
    bids_layout = BIDSLayout(BIDS_DIR, validate=True)
    masks_folder = MASKS_DIR
    output_path = "out"

    f = io.StringIO()
    with redirect_stdout(f):
        run_svrtk(
            sub,
            config_sub,
            CONFIG_PATH,
            bids_layout,
            BIDS_DIR,
            output_path,
            masks_folder,
            participant_label=None,
            fake_run=True,
        )
    out = f.getvalue()
    print(out)
    with open(FILE_DIR / "output/svrtk_out.txt") as f:
        svrtk = f.readlines()
    svrtk = "".join(svrtk)

    assert out == svrtk
