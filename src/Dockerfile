FROM arm64v8/ubuntu:20.04 as tvmc

ENV DEBIAN_FRONTEND noninteractive

# LLVM
ARG LLVM_VERSION="llvmorg-14.0.6"
ARG LLVM_BIN="clang+llvm-14.0.6-aarch64-linux-gnu"

# TVM
ARG TVM_VERSION="v0.9.0"

# DLPACK - fix due to build errors with newest dlpack
ARG DLPACK_VERSION="v0.5"

RUN set -e \
    ### Install packages
    && apt update \
    && apt install -y --no-install-recommends git wget python3 python3-pip python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev libncurses5 \
    ### Build tvm from source
    && cd /tmp; wget https://github.com/llvm/llvm-project/releases/download/${LLVM_VERSION}/${LLVM_BIN}.tar.xz \
    && mkdir llvm; tar -xf ${LLVM_BIN}.tar.xz -C /tmp/llvm \
    && git clone --recursive https://github.com/apache/tvm tvm \
    && cd tvm; git checkout tags/${TVM_VERSION} \
    && cd 3rdparty/dlpack; git checkout tags/${DLPACK_VERSION}; cd ../.. \
    && mkdir build \
    && sed -i "/set(USE_LLVM OFF)/c\set(USE_LLVM /tmp/llvm/${LLVM_BIN}/bin/llvm-config)" cmake/config.cmake \
    && cp cmake/config.cmake build \
    && cd build; cmake ..; make -j$(nproc); cd .. \
    && cd python; python3 setup.py install --user; cd .. \
    && pip3 install --user numpy Pillow decorator attrs typing-extensions onnx tornado psutil 'xgboost<1.6.0' cloudpickle \
    ### Clean-up
    && rm /tmp/${LLVM_BIN}.tar.xz \
    && rm -R /tmp/tvm \
    && apt clean all
