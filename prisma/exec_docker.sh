#!/bin/bash

# This script is used to run prisma on a docker container
# It takes as input the parameters to be passed to the prisma executable
# It is used by the run_11_nodes.sh script

PARAMS=$1
P=$(pwd)
echo "path = ${P}"
echo "parameters = ${PARAMS}"
docker run --rm \
    --runtime=nvidia --gpus all \
    --name test_session \
    -v "${P}/examples:/app/prisma/examples"\
    -p 16666:16666 \
    allicheredha/prismacopy:latest \
    $PARAMS