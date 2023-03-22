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
from .definitions import AUTO_MASK_PATH, PATTERN


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

    return build_path(ents, [bids_dir + PATTERN])


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
    return [str(run_dict[s]) for s in stacks], auto_masks


def filter_run_list(stacks, run_list):
    run_dict = find_run_id(run_list)
    return [str(run_dict[s]) for s in stacks]


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
    """Read through a CSV file and maps each row
    to the entry of a list.
    """
    file_list = []
    reader = csv.DictReader(open(csv_path))
    for i, line in enumerate(reader):
        file_list.append(line)
    return file_list


def print_title(text, center=True, char="-"):
    try:
        terminal_size = os.get_terminal_size().columns
    except Exception:
        terminal_size = 80
    char_length = min(len(text) + 10, terminal_size)
    chars = char * char_length
    text = text.upper()
    if center:
        chars = chars.center(terminal_size)
        text = text.center(terminal_size)
    print("\n" + chars + "\n" + text + "\n" + chars + "\n")


###########################################################
#   Iterate through a directory or a BIDS layout.
# 1. iter_dir: Iterate a BIDS-like directory and constructs
#       a dictionary indexed as dict[sub][ses] with the list
#       of paths to the corresponding folder.
# 2. iter_bids_dict: Returns an iterator over a dictionary
#       built with iter_dir
# 3. iter_bids: Iterate a BIDSLayout.
###########################################################


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
            for keyvalue in iter_bids_dict(value, _depth + 1, max_depth=max_depth):
                yield (key, *keyvalue)
        else:
            # If value is not dict type then yield the value
            yield (key, value)


def iter_bids(
    bids_layout,
    extension="nii.gz",
    datatype="anat",
    suffix="T2w",
    target=None,
    return_type="filename",
):
    """Return a single iterator over the BIDSLayout obtained from
    pybids - flexibly handles cases with and without a session date.
    """
    for sub in sorted(bids_layout.get_subjects()):
        for ses in [None] + sorted(bids_layout.get_sessions(subject=sub)):
            run_list = sorted(bids_layout.get_runs(subject=sub, session=ses))
            for run in [None] + run_list:
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
                for o in out:
                    yield (sub, ses, run, o)
