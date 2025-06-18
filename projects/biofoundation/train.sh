#!/bin/bash

# Run the python script with the specified command and pass any additional arguments
TF_CPP_MIN_LOG_LEVEL=3 python birdset/train.py --config-path '../projects/biofoundation/configs' --config-dir 'configs' --multirun 'logger=wandb_biofoundation' "$@"