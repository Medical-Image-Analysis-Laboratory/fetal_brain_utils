from fetal_brain_utils.cli.run_niftymic import (
    iterate_subject as run_niftymic,
    main as main_niftymic,
)
from fetal_brain_utils.cli.run_svrtk import (
    iterate_subject as run_svrtk,
    main as main_svrtk,
)

from fetal_brain_utils.cli.run_nesvor_from_config import (
    iterate_subject as run_nesvor,
    main as main_nesvor,
)
import json
from bids import BIDSLayout
from pathlib import Path
import pytest

FILE_DIR = Path(__file__).parent.resolve()
BIDS_DIR = FILE_DIR / "data"
CONFIG_PATH = FILE_DIR / "data/code/params.json"
MASKS_DIR = FILE_DIR / "data/derivatives/masks"


def read_and_replace_txt(file_path):
    with open(file_path) as f:
        txt = f.readlines()
    return "".join(txt).replace("<PATH>", str(FILE_DIR))


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
    niftymic = read_and_replace_txt(FILE_DIR / "output/niftymic_out.txt")

    assert captured.out == niftymic


def remove_blanks(str):
    return str.replace("\n", "").replace(" ", "")


def test_niftymic_interface(capsys):
    with pytest.raises(SystemExit):
        main_niftymic(["-h"])
    captured = capsys.readouterr()
    niftymic = read_and_replace_txt(FILE_DIR / "output/niftymic_main_help.txt")
    assert remove_blanks(captured.out) == remove_blanks(niftymic)


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
    svrtk = read_and_replace_txt(FILE_DIR / "output/svrtk_out.txt")
    assert captured.out == svrtk


def test_svrtk_interface(capsys):
    with pytest.raises(SystemExit):
        main_svrtk(["-h"])
    captured = capsys.readouterr()
    svrtk = read_and_replace_txt(FILE_DIR / "output/svrtk_main_help.txt")
    assert remove_blanks(captured.out) == remove_blanks(svrtk)


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
    nesvor = read_and_replace_txt(FILE_DIR / "output/nesvor_source_out.txt")
    assert captured.out == nesvor


def test_nesvor_source_interface(capsys):
    with pytest.raises(SystemExit):
        main_nesvor(["-h"])
    captured = capsys.readouterr()
    nesvor = read_and_replace_txt(FILE_DIR / "output/nesvor_source_main_help.txt")
    assert remove_blanks(captured.out) == remove_blanks(nesvor)
