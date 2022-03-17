#!/bin/sh

apt-get install gcc g++ python

apt-get install libzmq5 libzmq5-dev
apt-get install libprotobuf-dev
apt-get install protobuf-compiler

cd my_network/
cd ns3gym/
sh compile_proto.sh