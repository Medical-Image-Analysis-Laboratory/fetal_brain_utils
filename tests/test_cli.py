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

from fetal_brain_utils.cli.run_nesvor_docker import (
    iterate_subject as run_nesvor_docker,
    main as main_nesvor_docker,
)
from fetal_brain_utils.cli.run_mialsrtk import (
    main as main_mialsrtk,
)
import json
from bids import BIDSLayout
from pathlib import Path
import pytest
from fetal_brain_utils.utils import iter_dir

FILE_DIR = Path(__file__).parent.resolve()
BIDS_DIR = FILE_DIR / "data"
CONFIG_PATH = FILE_DIR / "data/code/params.json"
MASKS_DIR = FILE_DIR / "data/derivatives/masks"


def read_and_replace_txt(file_path):
    with open(file_path) as f:
        txt = f.readlines()
    return "".join(txt).replace("<PATH>", str(FILE_DIR))


def remove_blanks(str):
    return str.replace("\n", "").replace(" ", "")


def test_mialsrtk(capsys):
    """Test the text output of run_mialsrtk."""
    main_mialsrtk(
        [
            "--data_path",
            str(BIDS_DIR),
            "--param_file",
            str(CONFIG_PATH),
            "--masks_derivatives_dir",
            str(MASKS_DIR),
            "--out_folder",
            "out",
            "--fake_run",
        ]
    )
    captured = capsys.readouterr()
    mialsrtk = read_and_replace_txt(FILE_DIR / "output/mialsrtk_main.txt")
    print(f"{captured.out}")
    assert captured.out == mialsrtk


def test_mialsrtk_interface(capsys):
    with pytest.raises(SystemExit):
        main_mialsrtk(["-h"])
    captured = capsys.readouterr()
    mialsrtk = read_and_replace_txt(FILE_DIR / "output/mialsrtk_main_help.txt")
    assert remove_blanks(captured.out) == remove_blanks(mialsrtk)


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


def test_nesvor_docker_iterate_subject(capsys):
    """Test the text output of run_nesvor_from_config."""
    with open(CONFIG_PATH, "r") as f:
        params = json.load(f)
    sub = "simu005"
    config_sub = params[sub]
    sub_ses_dict = iter_dir(BIDS_DIR, add_run_only=True)
    masks_folder = MASKS_DIR
    output_path = "out"

    run_nesvor_docker(
        sub,
        config_sub,
        sub_ses_dict,
        BIDS_DIR,
        output_path,
        masks_folder,
        participant_label=None,
        target_res=[1.1],
        config=params,
        fake_run=True,
    )
    captured = capsys.readouterr()
    nesvor = read_and_replace_txt(FILE_DIR / "output/nesvor_docker_out.txt")
    assert captured.out == nesvor


def test_nesvor_docker_interface(capsys):
    with pytest.raises(SystemExit):
        main_nesvor_docker(["-h"])
    captured = capsys.readouterr()
    nesvor = read_and_replace_txt(FILE_DIR / "output/nesvor_docker_main_help.txt")
    assert remove_blanks(captured.out) == remove_blanks(nesvor)
