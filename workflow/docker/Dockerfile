FROM ubuntu:18.04
ARG DEBIAN_FRONTEND=noninteractive

# install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    swig \
    wget \
    virtualenv \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libc6-dev \
    libosmesa6-dev \
    patchelf \
    software-properties-common \
    gfortran \
    && apt-get install --no-install-recommends -y file g++ gcc make gdb strace ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get --no-install-recommends install -y python3.8-dev python3.8 python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install MPI
RUN wget -q http://www.mpich.org/static/downloads/3.1.4/mpich-3.1.4.tar.gz
RUN tar xf mpich-3.1.4.tar.gz
WORKDIR mpich-3.1.4
RUN ./configure --disable-fortran --enable-fast=all,O3 --prefix=/usr 
RUN make -j$(nproc)
RUN make install 
RUN ldconfig
WORKDIR /

# setup env
ENV VIRTUAL_ENV=/venv
RUN virtualenv --python=python3.8 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install python packages
# RUN pip install sbi \
RUN pip install numpy \
    && pip install h5py \
    && pip install scipy \
    && pip install numba \
    # && pip install matplotlib \
    && pip install tqdm \
    && pip install mpi4py
    # && pip install torch

COPY . .

WORKDIR /

# cleanup 
RUN pip cache purge

# set the default command to start a shell
CMD ["/bin/bash"]