![fetal_brain_utils](https://github.com/Medical-Image-Analysis-Laboratory/fetal_brain_utils/actions/workflows/python.yml/badge.svg)
# Readme
This repository contains various helper functions as well as wrappers to run some common super-resolution pipelines for fetal brain MRI with [BIDS](https://bids.neuroimaging.io/)-formatted input and output.

It contains some command-line commands that are listed below, as well as utility functions for cropping nifti images and iterating BIDS directories.

## Commands:
`run_mialsrtk`
```
usage: run_mialsrtk [-h] --data_path DATA_PATH --masks_path MASKS_PATH --config CONFIG --out_path OUT_PATH [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]] [--fake_run]
                    [--docker_version DOCKER_VERSION] [--run_type {sr,preprocessing}] [--automated] [--txt_to TXT_TO] [--labels_derivatives_dir LABELS_DERIVATIVES_DIR]
                    [--pymialsrtk_path PYMIALSRTK_PATH] [--verbose] [--no_python_mount] [--complement_missing_masks]

Parser for wrapper script of MIALSRTK This requires to provide data in a BIDS format.

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to the data. (default: None)
  --masks_path MASKS_PATH
                        Path to the brain masks. (default: None)
  --config CONFIG       Path to the configuration file. (default: None)
  --out_path OUT_PATH   Where the results are stored. (default: None)
  --participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]
                        The label(s) of the participant(s) that should be analyzed. (default: None)
  --fake_run            Whether to only print the commands instead of running them (default: False)
  --docker_version DOCKER_VERSION
                        Docker version of the pipeline used. (default: v2.1.0-dev)
  --run_type {sr,preprocessing}
                        Type of pipeline that is run. Can choose between running the super-resolution pipeline (`sr`) or only preprocesing (`preprocessing`). (default: sr)
  --automated           Run with automated masks (default: False)
  --txt_to TXT_TO       Where the text output is stored. By default, it is output to the command line. (default: None)
  --labels_derivatives_dir LABELS_DERIVATIVES_DIR
                        Where the labels are stored (absolute path). (default: None)
  --pymialsrtk_path PYMIALSRTK_PATH
                        Where pymialsrtk is located. (default: /home/tsanchez/Documents/mial/repositories/mialsuperresolutiontoolkit/pymialsrtk)
  --verbose             Verbose output (default: False)
  --no_python_mount     Whether the python folder should not be mounted. (default: True)
  --complement_missing_masks
                        Whether missing masks should be replaced with automated masks. (default: False)


```

`run_niftymic` 
```
usage: run_niftymic [-h] --data_path DATA_PATH --masks_path MASKS_PATH --config CONFIG --out_path OUT_PATH [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]] [--fake_run] [--alpha ALPHA]
                    [--nprocs NPROCS] [--use_preprocessed]

Parser for wrapper script of NiftyMIC This requires to provide data in a BIDS format.

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to the data. (default: None)
  --masks_path MASKS_PATH
                        Path to the brain masks. (default: None)
  --config CONFIG       Path to the configuration file. (default: None)
  --out_path OUT_PATH   Where the results are stored. (default: None)
  --participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]
                        The label(s) of the participant(s) that should be analyzed. (default: None)
  --fake_run            Whether to only print the commands instead of running them (default: False)
  --alpha ALPHA         Alpha to be used. (default: 0.068)
  --nprocs NPROCS       Number of processes used in parallel (default: 1)
  --use_preprocessed    Whether the parameter study should use bias corrected images as input. (default: False)
```

`run_svrtk`
```
usage: run_svrtk [-h] --data_path DATA_PATH --masks_path MASKS_PATH --config CONFIG --out_path OUT_PATH [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]] [--fake_run]

Parser for wrapper script of SVRTK This requires to provide data in a BIDS format.

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to the data. (default: None)
  --masks_path MASKS_PATH
                        Path to the brain masks. (default: None)
  --config CONFIG       Path to the configuration file. (default: None)
  --out_path OUT_PATH   Where the results are stored. (default: None)
  --participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]
                        The label(s) of the participant(s) that should be analyzed. (default: None)
  --fake_run            Whether to only print the commands instead of running them (default: False)

```

`run_nesvor`
```
usage: run_nesvor [-h] --data_path DATA_PATH --masks_path MASKS_PATH --config CONFIG --out_path OUT_PATH [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]] [--fake_run] --target_res
                         TARGET_RES [TARGET_RES ...]

Parser for wrapper script of NeSVoR (docker) This requires to provide data in a BIDS format.

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to the data. (default: None)
  --masks_path MASKS_PATH
                        Path to the brain masks. (default: None)
  --config CONFIG       Path to the configuration file. (default: None)
  --out_path OUT_PATH   Where the results are stored. (default: None)
  --participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]
                        The label(s) of the participant(s) that should be analyzed. (default: None)
  --fake_run            Whether to only print the commands instead of running them (default: False)
  --target_res TARGET_RES [TARGET_RES ...]
                        Target resolutions at which the reconstruction should be done. (default: None)
```

`run_nesvor_source`
```
usage: run_nesvor_source [-h] --data_path DATA_PATH --masks_path MASKS_PATH --config CONFIG --out_path OUT_PATH [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]] [--fake_run] --target_res
                         TARGET_RES [TARGET_RES ...] [--single_precision]

Parser for wrapper script of NeSVoR (source) This requires to provide data in a BIDS format.

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to the data. (default: None)
  --masks_path MASKS_PATH
                        Path to the brain masks. (default: None)
  --config CONFIG       Path to the configuration file. (default: None)
  --out_path OUT_PATH   Where the results are stored. (default: None)
  --participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]
                        The label(s) of the participant(s) that should be analyzed. (default: None)
  --fake_run            Whether to only print the commands instead of running them (default: False)
  --target_res TARGET_RES [TARGET_RES ...]
                        Target resolutions at which the reconstruction should be done. (default: None)
  --single_precision    Whether single precision should be used for training (by default, half precision is used.) (default: False)
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

## ToDos:
- [ ] Change name of `visualize_sr` to `visualize` + add dependency to `install_requires`
- [x] Change `run_nesvor_docker` to `run_nesvor`
- [ ] Simplify the interface `run_mialsrtk` and set python mounting to *off* by default. Have some control on whether the path is found.
- [ ] Harmonize BIDS directory reading to `pybids` only.
- [ ] Publish the repository (select a relevant license.)

