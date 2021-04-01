FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ENV TORCHVER=v1.8.1
ENV VISIONVER=v0.9.1
ENV AUDIOVER=v0.8.1

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies (1)
RUN apt-get update && apt-get install -y \
        automake autoconf libpng-dev nano wget npm \
        curl zip unzip libtool swig zlib1g-dev pkg-config \
        git wget xz-utils python3-mock libpython3-dev \
        libpython3-all-dev python3-pip g++ gcc make \
        pciutils cpio gosu git liblapack-dev liblapacke-dev

# Install dependencies (2)
RUN pip3 install --upgrade pip \
    && pip3 install --upgrade onnx \
    && pip3 install --upgrade onnxruntime \
    && pip3 install --upgrade gdown \
    && pip3 install cmake==3.18.4 \
    && pip3 install --upgrade pyyaml \
    && pip3 install --upgrade ninja \
    && pip3 install --upgrade yapf \
    && pip3 install --upgrade six \
    && pip3 install --upgrade wheel \
    && pip3 install --upgrade moc \
    && ldconfig

# Build
#   Bypass CUDA and CUB version checks.
RUN git clone -b ${TORCHVER} --recursive https://github.com/pytorch/pytorch
RUN cd /pytorch \
    && sed -i -e "/^#ifndef THRUST_IGNORE_CUB_VERSION_CHECK$/i #define THRUST_IGNORE_CUB_VERSION_CHECK" \
                 /usr/local/cuda/targets/x86_64-linux/include/thrust/system/cuda/config.h \
    && cat /usr/local/cuda/targets/x86_64-linux/include/thrust/system/cuda/config.h \
    && sed -i -e "/^if(DEFINED GLIBCXX_USE_CXX11_ABI)/i set(GLIBCXX_USE_CXX11_ABI 1)" \
                 CMakeLists.txt \
    && pip3 install -r requirements.txt \
    && USE_NCCL=OFF python3 setup.py build \
    && python3 setup.py bdist_wheel

RUN git clone -b ${VISIONVER} https://github.com/pytorch/vision.git
RUN cd /vision \
    && git submodule update --init --recursive \
    && pip3 install /pytorch/dist/*.whl \
    && python3 setup.py build \
    && python3 setup.py bdist_wheel

RUN git clone -b ${AUDIOVER} https://github.com/pytorch/audio.git
RUN cd /audio \
    && git submodule update --init --recursive \
    && apt-get install -y sox libsox-dev \
    && python3 setup.py build \
    && python3 setup.py bdist_wheel

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

WORKDIR /workspace
