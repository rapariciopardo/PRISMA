#!/bin/bash

# module load singularity/3.5.2

# singularity exec --nv prisma_latest.sif python3 main.py --simTime=1 --save_models=0
for ((i=0; i<5; i++))
do
    echo "module load singularity 1111"
done