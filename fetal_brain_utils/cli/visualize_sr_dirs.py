from tkinter import ttk
import tkinter as tk
import os
import argparse
from fetal_brain_utils.utils import iter_bids
from functools import partial
from bids import BIDSLayout
from collections import defaultdict
from functools import reduce


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
    for run in iter_bids(
        layout_list[0]
    ):  # Currently, it will look at SR with run-ids, but we can skip it also with skip_run=True and return_all=True. I should make something that is more generic to handle this.
        sub, ses, _, path = run
        if (sub, ses) not in sub_ses_list:
            sub_ses_list.append((sub, ses))
        sub_ses = f"{sub} - {ses}"
        path = [path] if not isinstance(path, list) else path
        for p in path:
            sub_ses_dict[sub_ses].append(p)
            name_dict[sub_ses].append(os.path.basename(layout_list[0].root))

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
                name_dict[sub_ses].append(os.path.basename(l.root))
    return sub_ses_dict, name_dict


def main():
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

    args = parser.parse_args()
    # Without validation, the BIDSLayout will also list down all folders
    # (i.e. for niftymic, where we have a folder with a structure niftymic/[niftymic, chuv003, chuv004],
    # it will go all the way down niftymic, which should be ignored. However, with validation=True, it
    # returns nothing because the file_names are not BIDS compliant.)
    layout_list = [BIDSLayout(dir, validate=False) for dir in args.bids_dir]
    sub_ses_dict = defaultdict(list)
    name_dict = defaultdict(list)

    sub_ses_dict, name_dict = build_sub_ses_dict(layout_list)
    values = list(sub_ses_dict.keys())
    main_window = tk.Tk()
    main_window.config(width=300, height=200)
    main_window.title("Explore BIDS dataset")

    combo = ttk.Combobox(values=values)

    def selection_changed(event, files, namelist=None):
        selection = combo.get()
        cmd = "itksnap "
        print("Displaying:")
        for i, file in enumerate(files[selection]):
            cmd += f"-g {file} -o " if i == 0 else f"{file} "
            print(f"\t{file}")
        print(cmd)
        os.system(cmd)

    combo.bind(
        "<<ComboboxSelected>>",
        partial(selection_changed, files=sub_ses_dict, namelist=name_dict),
    )
    combo.place(x=50, y=50)
    main_window.mainloop()


if __name__ == "__main__":
    main()
