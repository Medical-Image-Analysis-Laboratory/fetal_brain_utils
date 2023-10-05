from pathlib import Path
from bids.layout import BIDSLayout
import argparse
import re
import pandas as pd
import nibabel as ni
import numpy as np


def iter_bids_custom(
    bids_layout,
    extension="nii.gz",
    datatype="anat",
    suffix="sigma",
    return_type="filename",
):
    for sub in sorted(bids_layout.get_subjects()):
        for ses in [None] + sorted(bids_layout.get_sessions(subject=sub)):
            use_suffix = suffix if suffix != "sigma_var" else "var"

            out = bids_layout.get(
                subject=sub,
                session=ses,
                extension=extension,
                datatype=datatype,
                suffix=use_suffix,
                return_type=return_type,
            )
            if suffix == "sigma_var":
                out = [o for o in out if "sigma" in o]
            yield (sub, ses, out)


def main(argv=None):
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate measures of uncertainty for the LR inputs given to NeSVoR.",
    )
    p.add_argument(
        "--data_path",
        default=None,
        required=True,
        help="Path to the data.",
    )

    p.add_argument(
        "--out_csv",
        default=None,
        required=True,
        help="Out path for the data.",
    )

    p.add_argument(
        "--normalize_subject",
        action="store_true",
        help="Normalize the data by subject.",
        default=False,
    )
    args = p.parse_args(argv)

    data_path = Path(args.data_path).resolve()
    out_csv = Path(args.out_csv).resolve()

    # Load a dictionary of subject-session-paths
    # sub_ses_dict = iter_dir(data_path, add_run_only=True)
    bids_layout = BIDSLayout(data_path, validate=False)
    # Create a variety of measurements based on the
    # sigma, var, and sigma_var maps.
    # They can be aggregated based on their mean, median, variance or
    # mad (median absolute deviation)
    df = pd.DataFrame(
        columns=[
            "sub",
            "ses",
            "run",
            "run_rec",
            "sigma_avg",
            "var_avg",
            "sigma_var_avg",
            "n_qc",
            "sigma_med",
            "var_med",
            "sigma_var_med",
            "sigma__var",
            "var__var",
            "sigma_var__var",
            "sigma_mad",
            "var_mad",
            "sigma_var_mad",
            "sigma_avg_s",
            "var_avg_s",
            "sigma_var_avg_s",
            "n_qc_s",
            "sigma_med_s",
            "var_med_s",
            "sigma_var_med_s",
            "sigma__var_s",
            "var__var_s",
            "sigma_var__var_s",
            "sigma_mad_s",
            "var_mad_s",
            "sigma_var_mad_s",
        ]
    )

    for sub, ses, out in iter_bids_custom(bids_layout, suffix="sigma"):
        for rec_run in [0, 1]:
            out_rec = [o for o in out if f"slices_run-{rec_run}" in o]
            if len(out_rec) == 0:
                continue
            ids = [int(re.search(r"_run-(\d+)_", element).group(1)) for element in out_rec]

            for i in range(max(ids)):
                if i not in ids:
                    continue

                out_id = [el for id_, el in zip(ids, out_rec) if id_ == i]
                sigma_tot = []
                var_tot = []
                sigma_var_tot = []
                # Aggregate the result of each slice counting equally
                sigma_tot_slice = []
                var_tot_slice = []
                sigma_var_tot_slice = []
                max_sigma = 0.0
                max_var = 0.0
                max_sigma_var = 0.0

                if args.normalize_subject:
                    for el in out_id:
                        max_sigma = max(ni.load(el).get_fdata().max(), max_sigma)
                        max_var = max(
                            ni.load(el.replace("_sigma", "_var")).get_fdata().max(), max_var
                        )
                        max_sigma_var = max(
                            ni.load(el.replace("_sigma", "_sigma_var")).get_fdata().max(),
                            max_sigma_var,
                        )
                aggr_fc = np.mean
                for el in out_id:
                    slice_sigma = ni.load(el).get_fdata()
                    slice_T2w = ni.load(el.replace("_sigma", "_T2w")).get_fdata()
                    slice_var = ni.load(el.replace("_sigma", "_var")).get_fdata()
                    slice_sigma_var = ni.load(el.replace("_sigma", "_sigma_var")).get_fdata()
                    loc = slice_T2w > 0
                    sigma = slice_sigma[loc]
                    var = slice_var[loc]
                    sigma_var = slice_sigma_var[loc]
                    # print(
                    #     f"{max_sigma:.3f} ({slice_sigma.max():3f}) "
                    #     f"{max_var:.3f} ({slice_var.max():3f}) "
                    #     f"{max_sigma_var:.3f} ({slice_sigma_var.max():3f})"
                    # )
                    if len(sigma) == 0:
                        print("Empty slice ", Path(el).name)
                        continue
                    if args.normalize_subject:
                        sigma /= max_sigma
                        var /= max_var
                        sigma_var /= max_sigma_var
                    sigma_tot += sigma.tolist()
                    var_tot += var.tolist()
                    sigma_var_tot += sigma_var.tolist()

                    sigma_tot_slice.append(aggr_fc(sigma))
                    var_tot_slice.append(aggr_fc(var))
                    sigma_var_tot_slice.append(aggr_fc(sigma_var))

                def mad(x):
                    return np.median(np.abs(x - np.median(x)))

                d = {
                    "sub": sub,
                    "ses": ses,
                    "run": i,
                    "run_rec": rec_run,
                    "sigma_avg": np.mean(sigma_tot),
                    "var_avg": np.mean(var_tot),
                    "sigma_var_avg": np.mean(sigma_var_tot),
                    "n_qc": len(sigma_tot),
                    "sigma_med": np.median(sigma_tot),
                    "var_med": np.median(var_tot),
                    "sigma_var_med": np.median(sigma_var_tot),
                    "sigma__var": np.var(sigma_tot),
                    "var__var": np.var(var_tot),
                    "sigma_var__var": np.var(sigma_var_tot),
                    "sigma_mad": mad(sigma_tot),
                    "var_mad": mad(var_tot),
                    "sigma_var_mad": mad(sigma_var_tot),
                    "sigma_avg_s": np.mean(sigma_tot_slice),
                    "var_avg_s": np.mean(var_tot_slice),
                    "sigma_var_avg_s": np.mean(sigma_var_tot_slice),
                    "n_qc_s": len(sigma_tot_slice),
                    "sigma_med_s": np.median(sigma_tot_slice),
                    "var_med_s": np.median(var_tot_slice),
                    "sigma_var_med_s": np.median(sigma_var_tot_slice),
                    "sigma__var_s": np.var(sigma_tot_slice),
                    "var__var_s": np.var(var_tot_slice),
                    "sigma_var__var_s": np.var(sigma_var_tot_slice),
                    "sigma_mad_s": mad(sigma_tot_slice),
                    "var_mad_s": mad(var_tot_slice),
                    "sigma_var_mad_s": mad(sigma_var_tot_slice),
                }
                print(d)
                # Add d to the dataframe:
                df = pd.concat([df, pd.DataFrame(d, index=[0])], ignore_index=True)

    # Check if the folder of out_csv exists, if not create it
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
