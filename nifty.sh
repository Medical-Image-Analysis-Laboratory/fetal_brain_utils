#!/bin/bash

PARAMETER=${1:-007}

run_niftymic \
--data_path ../../images_project/data_chuv_paul_part2/ \
--masks_folder ../../images_project/data_chuv_paul_part2/derivatives/masks/ \
--out_path ../../out/ \
--config config.json \
--participant_label chuv$PARAMETER
