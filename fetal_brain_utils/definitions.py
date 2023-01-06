AUTO_MASK_PATH = "/media/tsanchez/tsanchez_data/data/out_anon/masks"
OUT_JSON_ORDER = [
    "sr-id",
    "session",
    "ga",
    "stacks",
    "use_auto_mask",
    "config_path",
    "info",
    "im_path",
    "mask_path",
]
SLICER_PATH = "/home/tsanchez/Slicer-5.0.3-linux-amd64/Slicer"
PATTERN = (
    "/sub-{subject}[/ses-{session}][/{datatype}]/sub-{subject}"
    "[_ses-{session}][_acq-{acquisition}][_run-{run}]_{suffix}.{extension}"
)
