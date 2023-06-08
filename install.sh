#!/bin/sh
echo "Installing dependencies..."
apt-get update && apt-get install -y gcc g++ bc rsync libzmq5 libzmq3-dev libprotobuf-dev protobuf-compiler

echo "Installing python packages..."
python3 -m pip install --user --no-cache-dir --upgrade -r requirements.txt

echo "Compiling protobuf..."
cd prisma/ns3gym/
sh compile_proto.sh
