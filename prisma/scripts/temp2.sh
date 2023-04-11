#!/bin/bash

module load singularity/3.5.2
echo "runing singular"
singularity exec --nv prisma_latest.sif python3 temp1.py
