#!/bin/bash

# Datasets to loop through
#("beans_watkins" "beans_bats" "beans_cbi" "beans_dogs" "beans_humbugdb")
dnames=("beans_watkins" "beans_cbi" "beans_dogs" "beans_humbugdb" "beans_bats")
dclasses=(31 264 10 14 10)

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
  echo "No model provided: Example: Perch"
  exit 1
fi
# Loop through provided experiments
for experiment in "$@"; do
  echo "Running experiment $experiment"
  for i in "${!dnames[@]}"; do
    dname=${dnames[$i]}
    dclass=${dclasses[$i]}
    echo "Running with dataset_name=$dname and n_classes=$dclass"
    python birdset/train.py --multirun experiment="local/embedding/BEANS/$experiment" seed=1,2,3 datamodule.average=False datamodule.k_samples=0 datamodule.average=False logger.wandb.group="results_run9" trainer.devices=[2] trainer.min_epochs=1 trainer.max_epochs=50 datamodule.dataset.dataset_name=$dname datamodule.dataset.hf_path="DBD-research-group/$dname" datamodule.dataset.n_classes=$dclass
  done
done