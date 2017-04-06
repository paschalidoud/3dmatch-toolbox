#!/bin/bash

export PATH=$PATH:/is/software/nvidia/cuda-8.0/bin

if uname | grep -q Darwin; then
  CUDA_LIB_DIR=/usr/local/cuda/lib
  CUDNN_LIB_DIR=/usr/local/cudnn/v5.1/lib
elif uname | grep -q Linux; then
  CUDA_LIB_DIR=/is/software/nvidia/cuda-8.0/lib64
  CUDNN_LIB_DIR=/is/software/nvidia/cudnn-5.1/lib64
fi

CUDNN_INC_DIR=/is/software/nvidia/cudnn-5.1/include

# if use opencv, add this into the command line
# `pkg-config --cflags --libs opencv`

nvcc -std=c++11 -O3 -o demo demo.cu -I/is/software/nvidia/cuda-8.0/include -I$CUDNN_INC_DIR -L$CUDA_LIB_DIR -L$CUDNN_LIB_DIR -lcudart -lcublas -lcudnn -lcurand -D_MWAITXINTRIN_H_INCLUDED `pkg-config --cflags --libs opencv`
