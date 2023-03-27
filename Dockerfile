FROM tensorflow/tensorflow:2.8.3-gpu

# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
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



