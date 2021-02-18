# PyTorch-build
Provide Docker build sequences of PyTorch for various environments.  
**https://github.com/pytorch/pytorch**

## 1. Docker Image Environment
- Ubuntu 18.04 x86_64
- CUDA 11.0
- cuDNN 8.0
- PyTorch v1.7.1 (Build from source code. It will be downloaded automatically during docker build.)
- TorchVision v0.8.2
- TorchAudio v0.7.2

## 2. Usage - Docker Build
You can customize the Dockerfile to build and run your own container images on your own.
```bash
$ version=1.7.1
$ git clone -b ${version} https://github.com/PINTO0309/PyTorch-build.git
$ cd PyTorch-build
$ docker build -t pinto0309/pytorch-build:latest .

$ docker run --gpus all -it --rm \
    -v `pwd`:/workspace \
    -e LOCAL_UID=$(id -u $USER) \
    -e LOCAL_GID=$(id -g $USER) \
    pinto0309/pytorch-build:latest bash
```

## 3. Usage - Docker Pull / Run
You can download and run a pre-built container image from Docker Hub.
```bash
$ docker run --gpus all -it --rm \
    -v `pwd`:/workspace \
    -e LOCAL_UID=$(id -u $USER) \
    -e LOCAL_GID=$(id -g $USER) \
    pinto0309/pytorch-build:latest bash
```