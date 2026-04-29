#!/bin/bash

ENV_NAME='yolotest_env'

UPDATE_REQUIREMENTS=true

ml Python
ml Qt6

if [ -d "$ENV_NAME/bin" ]
then
    # The /my-dir directory exists, so print a message
    echo "The vevn already exists."
    if [ $UPDATE_REQUIREMENTS ]
    then
    echo "Updating requirements"
    pip freeze | grep -E '^[A-Za-z0-9_.-]+==[0-9a-zA-Z.+!-]+' | grep -v myquota > requirements.txt
    fi
else
    # The /my-dir directory does not exist, so print a message and create it
    echo "The venv does not exist. Creating it now..."
    python -m venv $ENV_NAME
    pip install -r requirements.txt
    pip install ultralytics
    python -m pip install yt-dlp
fi

source $ENV_NAME/bin/activate

export dataset_dir=$(pwd)/yolodir/datasets
export weights_dir=$(pwd)/yolodir/weights
mkdir -p dataset_dir
mkdir -p weights_dir

YOLO_CONFIG_DIR=$(pwd)/yolodir/config
export YOLO_CONFIG_DIR

# Update multiple serttings at once  
# see https://docs.ultralytics.com/quickstart/#modifying-settings
yolo settings datasets_dir=$dataset_dir weights_dir=$weights_dir runs_dir=$(pwd)/yolodir/runs
