from tkinter import ttk
import tkinter as tk
import os
import argparse
from fetal_brain_qc.utils import csv_to_list, iter_bids


from functools import partial
from bids import BIDSLayout

SLICER_PATH = "/home/tsanchez/Slicer-5.0.3-linux-amd64/Slicer"
# SLICER_PATH = "/Applications/Slicer.app/Contents/MacOS/Slicer"
from collections import defaultdict
from functools import reduce
import json
import operator
from pathlib import Path


def selection_changed(event, files, namelist=None):
    selection = combo.get()
    # cmd = f'{SLICER_PATH} --python-code "'
    cmd = "itksnap "
    for i, file in enumerate(files[selection]):
        cmd += f"-g {file} -o " if i == 0 else f"{file} "

        # if namelist:
        #    name = f"{Path(file).stem[19:]} - {namelist[selection][i]}"
        #    cmd += f"slicer.util.loadVolume('{file}', {{'name':'{name}'}}); "
        # else:
        #    cmd += f"slicer.util.loadVolume('{file}');"
    # cmd += '"'
    print(cmd)
    os.system(cmd)


def get_subject_dict(bids_list):
    files_dict, masks_dict = defaultdict(list), defaultdict(list)
    for i, run in enumerate(bids_list):
        sub, ses, im, mask = run["sub"], run["ses"], run["im"], run["mask"]
        files_dict[f"{sub} - {ses}"].append(run["im"])
        masks_dict[f"{sub} - {ses}"].append(run["mask"])
    return files_dict, masks_dict


def build_sub_ses_dict(layout_list):

    sub_ses_dict = defaultdict(list)
    name_dict = defaultdict(list)

    # List of unique (sub,ses) pairs
    sub_ses_list = []
    # Iterate the first dict to list all relevant (sub,ses) pairs.
    for run in iter_bids(layout_list[0], skip_run=True, return_all=True):
        sub, ses, _, path = run
        if (sub, ses) not in sub_ses_list:
            sub_ses_list.append((sub, ses))
        sub_ses = f"{sub} - {ses}"
        path = [path] if not isinstance(path, list) else path
        for p in path:
            sub_ses_dict[sub_ses].append(p)
            name_dict[sub_ses].append(args.names[0])
    # Iterate through the rest of BIDS datasets using *only* the
    # sub,ses pairs indexed before.
    for sub, ses in sub_ses_list:
        sub_ses = f"{sub} - {ses}"
        for i, l in enumerate(layout_list[1:]):
            path = l.get(
                subject=sub,
                session=ses,
                extension="nii.gz",
                datatype="anat",
                suffix="T2w",
                target=None,
                return_type="filename",
            )
            for p in path:
                sub_ses_dict[sub_ses].append(p)
                name_dict[sub_ses].append(args.names[i + 1])
    return sub_ses_dict, name_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--bids_dir",
        required=True,
        nargs="+",
        help=(
            "Path to the SR folders to be listed. Note that the subjects and "
            "session in the *first* bids directory will define what will be displayed."
        ),
    )

    parser.add_argument(
        "--names",
        nargs="+",
        help="Display names for the current data.",
    )

    args = parser.parse_args()
    if args.names:
        assert len(args.names) == len(args.bids_dir)
    layout_list = [BIDSLayout(dir, validate=False) for dir in args.bids_dir]
    sub_ses_dict = defaultdict(list)
    name_dict = defaultdict(list)

    name_dict, sub_ses_dict = build_sub_ses_dict(layout_list)

    values = list(sub_ses_dict.keys())
    main_window = tk.Tk()
    main_window.config(width=300, height=200)
    main_window.title("Explore BIDS dataset")
    combo = ttk.Combobox(values=values)
    combo.bind(
        "<<ComboboxSelected>>",
        partial(selection_changed, files=sub_ses_dict, namelist=name_dict),
    )
    combo.place(x=50, y=50)
    main_window.mainloop()
