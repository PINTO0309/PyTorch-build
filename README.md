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

## 2. Default build parameters

```
-- 
-- ******** Summary ********
-- General:
--   CMake version         : 3.18.4
--   CMake command         : /usr/local/lib/python3.6/dist-packages/cmake/data/bin/cmake
--   System                : Linux
--   C++ compiler          : /usr/bin/c++
--   C++ compiler id       : GNU
--   C++ compiler version  : 7.5.0
--   BLAS                  : MKL
--   CXX flags             :  -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DUSE_VULKAN_WRAPPER -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow
--   Build type            : Release
--   Compile definitions   : ONNX_ML=1;ONNXIFI_ENABLE_EXT=1;ONNX_NAMESPACE=onnx_torch;HAVE_MMAP=1;_FILE_OFFSET_BITS=64;HAVE_SHM_OPEN=1;HAVE_SHM_UNLINK=1;HAVE_MALLOC_USABLE_SIZE=1;USE_EXTERNAL_MZCRC;MINIZ_DISABLE_ZIP_READER_CRC32_CHECKS
--   CMAKE_PREFIX_PATH     : /usr/lib/python3/dist-packages;/usr/local/cuda
--   CMAKE_INSTALL_PREFIX  : /pytorch/torch
-- 
--   TORCH_VERSION         : 1.7.0
--   CAFFE2_VERSION        : 1.7.0
--   BUILD_CAFFE2          : ON
--   BUILD_CAFFE2_OPS      : ON
--   BUILD_CAFFE2_MOBILE   : OFF
--   BUILD_STATIC_RUNTIME_BENCHMARK: OFF
--   BUILD_BINARY          : OFF
--   BUILD_CUSTOM_PROTOBUF : ON
--     Link local protobuf : ON
--   BUILD_DOCS            : OFF
--   BUILD_PYTHON          : True
--     Python version      : 3.6.9
--     Python executable   : /usr/bin/python3
--     Pythonlibs version  : 3.6.9
--     Python library      : /usr/lib/libpython3.6m.so.1.0
--     Python includes     : /usr/include/python3.6m
--     Python site-packages: lib/python3/dist-packages
--   BUILD_SHARED_LIBS     : ON
--   BUILD_TEST            : True
--   BUILD_JNI             : OFF
--   BUILD_MOBILE_AUTOGRAD : OFF
--   INTERN_BUILD_MOBILE   : 
--   USE_ASAN              : OFF
--   USE_CPP_CODE_COVERAGE : OFF
--   USE_CUDA              : ON
--     CUDA static link    : OFF
--     USE_CUDNN           : ON
--     CUDA version        : 11.0
--     cuDNN version       : 8.0.5
--     CUDA root directory : /usr/local/cuda
--     CUDA library        : /usr/local/cuda/lib64/stubs/libcuda.so
--     cudart library      : /usr/local/cuda/lib64/libcudart.so
--     cublas library      : /usr/local/cuda/lib64/libcublas.so
--     cufft library       : /usr/local/cuda/lib64/libcufft.so
--     curand library      : /usr/local/cuda/lib64/libcurand.so
--     cuDNN library       : /usr/lib/x86_64-linux-gnu/libcudnn.so
--     nvrtc               : /usr/local/cuda/lib64/libnvrtc.so
--     CUDA include path   : /usr/local/cuda/include
--     NVCC executable     : /usr/local/cuda/bin/nvcc
--     NVCC flags          : -Xfatbin;-compress-all;-DONNX_NAMESPACE=onnx_torch;-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_80,code=compute_80;-Xcudafe;--diag_suppress=cc_clobber_ignored;-Xcudafe;--diag_suppress=integer_sign_change;-Xcudafe;--diag_suppress=useless_using_declaration;-Xcudafe;--diag_suppress=set_but_not_used;-Xcudafe;--diag_suppress=field_without_dll_interface;-Xcudafe;--diag_suppress=base_class_has_different_dll_interface;-Xcudafe;--diag_suppress=dll_interface_conflict_none_assumed;-Xcudafe;--diag_suppress=dll_interface_conflict_dllexport_assumed;-Xcudafe;--diag_suppress=implicit_return_from_non_void_function;-Xcudafe;--diag_suppress=unsigned_compare_with_zero;-Xcudafe;--diag_suppress=declared_but_not_referenced;-Xcudafe;--diag_suppress=bad_friend_decl;-std=c++14;-Xcompiler;-fPIC;--expt-relaxed-constexpr;--expt-extended-lambda;-Wno-deprecated-gpu-targets;--expt-extended-lambda;-Xcompiler;-fPIC;-DCUDA_HAS_FP16=1;-D__CUDA_NO_HALF_OPERATORS__;-D__CUDA_NO_HALF_CONVERSIONS__;-D__CUDA_NO_HALF2_OPERATORS__
--     CUDA host compiler  : /usr/bin/cc
--     NVCC --device-c     : OFF
--     USE_TENSORRT        : OFF
--   USE_ROCM              : OFF
--   USE_EIGEN_FOR_BLAS    : ON
--   USE_FBGEMM            : ON
--     USE_FAKELOWP          : OFF
--   USE_FFMPEG            : OFF
--   USE_GFLAGS            : OFF
--   USE_GLOG              : OFF
--   USE_LEVELDB           : OFF
--   USE_LITE_PROTO        : OFF
--   USE_LMDB              : OFF
--   USE_METAL             : OFF
--   USE_MKL               : OFF
--   USE_MKLDNN            : ON
--   USE_MKLDNN_CBLAS      : OFF
--   USE_NCCL              : ON
--     USE_SYSTEM_NCCL     : OFF
--   USE_NNPACK            : ON
--   USE_NUMPY             : ON
--   USE_OBSERVERS         : ON
--   USE_OPENCL            : OFF
--   USE_OPENCV            : OFF
--   USE_OPENMP            : ON
--   USE_TBB               : OFF
--   USE_VULKAN            : OFF
--   USE_PROF              : OFF
--   USE_QNNPACK           : ON
--   USE_PYTORCH_QNNPACK   : ON
--   USE_REDIS             : OFF
--   USE_ROCKSDB           : OFF
--   USE_ZMQ               : OFF
--   USE_DISTRIBUTED       : ON
--     USE_MPI             : ON
--     USE_GLOO            : ON
--     USE_TENSORPIPE      : ON
--   Public Dependencies  : Threads::Threads;caffe2::mkldnn
--   Private Dependencies : pthreadpool;cpuinfo;qnnpack;pytorch_qnnpack;nnpack;XNNPACK;fbgemm;/usr/lib/x86_64-linux-gnu/libnuma.so;fp16;/usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so;/usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so;gloo;tensorpipe;aten_op_header_gen;foxi_loader;rt;fmt::fmt-header-only;gcc_s;gcc;dl
-- Configuring done
-- Generating done
-- Build files have been written to: /pytorch/build
```

## 3. Usage - Docker Build
You can customize the Dockerfile to build and run your own container images on your own.
```bash
$ version=1.7.1
$ git clone -b ${version} https://github.com/PINTO0309/PyTorch-build.git
$ cd PyTorch-build
$ docker build -t pinto0309/pytorch-build:11.0.3-cudnn8-devel-ubuntu18.04 .

$ docker run --gpus all -it --rm \
    -v `pwd`:/workspace \
    -e LOCAL_UID=$(id -u $USER) \
    -e LOCAL_GID=$(id -g $USER) \
    pinto0309/pytorch-build:11.0.3-cudnn8-devel-ubuntu18.04 bash
```

## 4. Usage - Docker Pull / Run
You can download and run a pre-built container image from Docker Hub.
```bash
$ docker run --gpus all -it --rm \
    -v `pwd`:/workspace \
    -e LOCAL_UID=$(id -u $USER) \
    -e LOCAL_GID=$(id -g $USER) \
    pinto0309/pytorch-build:11.0.3-cudnn8-devel-ubuntu18.04 bash
```