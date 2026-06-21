#!/bin/bash

module purge
module load aidl/pytorch/2.2.0-cuda12.1

source .venv/bin/activate
export PYTHONPATH="/home/2017018/jaffol01/lung_map:$PYTHONPATH"