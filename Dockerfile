FROM tensorflow/tensorflow:latest-gpu

# install dependencies
RUN apt-get update && apt-get install -y gcc g++ bc rsync libzmq5 libzmq5-dev libprotobuf-dev protobuf-compiler
# make python3 default
RUN ln -s /usr/bin/python3 /usr/bin/python



# copy files
COPY --chown=$whoami . /app/.

WORKDIR /app


# install python dependencies
RUN python3 -m pip install --user --no-cache-dir --upgrade -r requirements.txt

# compile protobuf
RUN cd prisma/ns3gym/ && ./compile_proto.sh && cd ../..


# copy ns3 files
COPY --chown=$whoami prisma/ns3 ns3-gym/scratch/prisma
COPY --chown=$whoami prisma/ns3_model/ipv4-interface.cc ns3-gym/src/internet/model/.
COPY --chown=$whoami prisma/ns3_model/ipv4-interface.h ns3-gym/src/internet/model/.

# compile ns3
RUN cd ns3-gym && ./waf configure 
RUN cd ns3-gym && ./waf build --disable-tests --disable-examples

# run prisma
WORKDIR /app/prisma
# ENTRYPOINT [ "python3", "main.py" ]



