# Readme
This repo is still much of a WIP.
- In particular, it uses various conventions for reading BIDS directories (csv listing, bidslayout, `sub_ses_dict`). This should be harmonized. 
- The outputs with mialsrtk, nesvor and niftymic are not consistently formatted. This should be revised. 
- This repository should be made public and incorporated as a dependency for other repositories like `fetal_brain_qc`.


## Commands:
`run_mialsrtk`
```
usage: run_mialsrtk [-h] [--data_path DATA_PATH] [--docker_version DOCKER_VERSION] [--run_type {sr,preprocessing}] [--automated] [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
                    [--txt_to TXT_TO] [--param_file PARAM_FILE] [--out_folder OUT_FOLDER] [--masks_derivatives_dir MASKS_DERIVATIVES_DIR] [--labels_derivatives_dir LABELS_DERIVATIVES_DIR]
                    [--pymialsrtk_path PYMIALSRTK_PATH] [--verbose] [--no_python_mount] [--complement_missing_masks]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path where the data are located (default: /media/tsanchez/tsanchez_data/data/data)
  --docker_version DOCKER_VERSION
                        Docker version of the pipeline used. (default: v2.1.0-dev)
  --run_type {sr,preprocessing}
                        Type of pipeline that is run. Can choose between running the super-resolution pipeline (`sr`) or only preprocesing (`preprocessing`). (default: sr)
  --automated           Run with automated masks (default: False)
  --participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]
                        The label(s) of the participant(s) that should be analyzed. (default: None)
  --txt_to TXT_TO       Where the text output is stored. By default, it is output to the command line. (default: None)
  --param_file PARAM_FILE
                        Where the json parameters are stored, relatively from code/ (default: None)
  --out_folder OUT_FOLDER
                        Where the results are stored. (default: None)
  --masks_derivatives_dir MASKS_DERIVATIVES_DIR
                        Where the masks are stored (absolute path). (default: None)
  --labels_derivatives_dir LABELS_DERIVATIVES_DIR
                        Where the labels are stored (absolute path). (default: None)
  --pymialsrtk_path PYMIALSRTK_PATH
                        Where pymialsrtk is located. (default: /home/tsanchez/Documents/mial/repositories/mialsuperresolutiontoolkit/pymialsrtk)
  --verbose             Verbose output (default: False)
  --no_python_mount     Whether the python folder should not be mounted. (default: False)
  --complement_missing_masks
                        Whether missing masks should be replaced with automated masks. (default: False)

```

`run_niftymic` 
```
usage: run_niftymic [-h] [--data_path DATA_PATH] --masks_folder MASKS_FOLDER --out_path OUT_PATH [--alpha ALPHA] --config CONFIG [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
                    [--nprocs NPROCS] [--use_preprocessed] [--fake_run]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path where the data are located (default: /media/tsanchez/tsanchez_data/data/data)
  --masks_folder MASKS_FOLDER
                        Folder where the masks are located. (default: None)
  --out_path OUT_PATH   Folder where the output will be stored. (default: None)
  --alpha ALPHA         Alpha to be used. (default: 0.068)
  --config CONFIG       Config path. (default: None)
  --participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]
                        Label of the participant (default: None)
  --nprocs NPROCS       Number of processes used in parallel (default: 1)
  --use_preprocessed    Whether the parameter study should use bias corrected images as input. (default: False)
  --fake_run            Whether to only print the commands instead of running them (default: False)
```


`run_nesvor_docker`
```
usage: run_nesvor_docker [-h] [--data_path DATA_PATH] --masks_folder MASKS_FOLDER --out_path OUT_PATH [--config CONFIG] [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]] --target_res
                         TARGET_RES [TARGET_RES ...]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path where the data are located (default: /media/tsanchez/tsanchez_data/data/data)
  --masks_folder MASKS_FOLDER
                        Folder where the masks are located. (default: None)
  --out_path OUT_PATH   Folder where the output will be stored. (default: None)
  --config CONFIG       Config path in data_path/code (default: `params.json`) (default: params.json)
  --participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]
                        Label of the participant (default: None)
  --target_res TARGET_RES [TARGET_RES ...]
                        Target resolutions at which the reconstruction should be done. (default: None)
```

`visualize_sr` - Useful to visualize and navigate repositories 
```
usage: visualize_sr [-h] --bids_dir BIDS_DIR [BIDS_DIR ...]

optional arguments:
  -h, --help            show this help message and exit
  --bids_dir BIDS_DIR [BIDS_DIR ...]
                        Path to the SR folders to be listed. Note that the subjects and session in the *first* bids directory will define what will be displayed. (default: None)
```

## Other utilities
`visualize_lr`
`run_nesvor`

## ToDos:
- Change name of `visualize_sr` to `visualize` + add dependency to `install_requires`
- Change `run_nesvor_docker` to `run_nesvor`
- Simplify the interface `run_mialsrtk` and set python mounting to *off* by default. Have some control on whether the path is found.