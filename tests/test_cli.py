from fetal_brain_utils.cli.run_niftymic import (
    iterate_subject as run_niftymic,
    main as main_niftymic,
)
from fetal_brain_utils.cli.run_svrtk import (
    iterate_subject as run_svrtk,
)

from fetal_brain_utils.cli.run_nesvor_from_config import (
    iterate_subject as run_nesvor,
)
import json
from bids import BIDSLayout
from pathlib import Path


FILE_DIR = Path(__file__).parent.resolve()
BIDS_DIR = FILE_DIR / "data"
CONFIG_PATH = FILE_DIR / "data/code/params.json"
MASKS_DIR = FILE_DIR / "data/derivatives/masks"


def test_niftymic_iterate_subject(capsys):
    """Test the text output of run_niftymic."""
    with open(CONFIG_PATH, "r") as f:
        params = json.load(f)
    sub = "simu005"
    config_sub = params[sub]
    bids_layout = BIDSLayout(BIDS_DIR, validate=True)
    masks_folder = MASKS_DIR
    output_path = "out"

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
    captured = capsys.readouterr()

    with open(FILE_DIR / "output/niftymic_out.txt") as f:
        niftymic = f.readlines()
    niftymic = "".join(niftymic).replace("<PATH>", str(FILE_DIR))

    assert captured.out == niftymic


def test_svrtk_iterate_subject(capsys):
    """Test the text output of run_svrtk."""
    with open(CONFIG_PATH, "r") as f:
        params = json.load(f)
    sub = "simu005"
    config_sub = params[sub]
    bids_layout = BIDSLayout(BIDS_DIR, validate=True)
    masks_folder = MASKS_DIR
    output_path = "out"

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
    captured = capsys.readouterr()
    with open(FILE_DIR / "output/svrtk_out.txt") as f:
        svrtk = f.readlines()
    svrtk = "".join(svrtk).replace("<PATH>", str(FILE_DIR))

    assert captured.out == svrtk


def test_nesvor_source_iterate_subject(capsys):
    """Test the text output of run_nesvor_from_config."""
    with open(CONFIG_PATH, "r") as f:
        params = json.load(f)
    sub = "simu005"
    config_sub = params[sub]
    bids_layout = BIDSLayout(BIDS_DIR, validate=True)
    masks_folder = MASKS_DIR
    output_path = "out"

    run_nesvor(
        sub,
        config_sub,
        bids_layout,
        BIDS_DIR,
        output_path,
        masks_folder,
        participant_label=None,
        target_res=[1.1],
        single_precision=False,
        config=params,
        fake_run=True,
    )
    captured = capsys.readouterr()
    with open(FILE_DIR / "output/nesvor_source_out.txt") as f:
        nesvor = f.readlines()
    nesvor = "".join(nesvor).replace("<PATH>", str(FILE_DIR))
    assert captured.out == nesvor
