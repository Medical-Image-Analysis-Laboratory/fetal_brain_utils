from tkinter import ttk
import tkinter as tk
import os
import argparse
from fetal_brain_utils.utils import csv_to_list
from fetal_brain_utils.definitions import SLICER_PATH
from collections import defaultdict
from functools import partial


def get_subject_dict(bids_list):
    files_dict, masks_dict = defaultdict(list), defaultdict(list)
    for i, run in enumerate(bids_list):
        sub, ses, im, mask = run["sub"], run["ses"], run["im"], run["mask"]
        files_dict[f"{sub} - {ses}"].append(im)
        masks_dict[f"{sub} - {ses}"].append(mask)
    return files_dict, masks_dict


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "bids_csv",
        help="Path to the BIDS csv file.",
    )

    args = parser.parse_args()
    bids_list = csv_to_list(args.bids_csv)

    files_dict, masks_dict = get_subject_dict(bids_list)
    values = list(files_dict.keys())
    main_window = tk.Tk()
    main_window.config(width=300, height=200)
    main_window.title("Explore BIDS dataset")
    combo = ttk.Combobox(values=values)

    def selection_changed(event, files, masks):
        selection = combo.get()

        cmd = f'{SLICER_PATH} --python-code "'
        for file, m in zip(files[selection], masks[selection]):
            cmd += f"slicer.util.loadVolume('{file}');"
            if "refined_mask" in m:
                cmd += f"slicer.util.loadLabelVolume('{m}');"
        cmd += 'slicer.util.setSliceViewerLayers(labelOpacity=0.5);"'
        print(cmd)
        os.system(cmd)

    combo.bind(
        "<<ComboboxSelected>>",
        partial(selection_changed, files=files_dict, masks=masks_dict),
    )
    combo.place(x=50, y=50)
    main_window.mainloop()


if __name__ == "__main__":
    main()
