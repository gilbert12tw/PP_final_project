CC = gcc
CXX = g++
NVCC = nvcc
HIPCC = hipcc

CXXFLAGS = -O3 -lm -funroll-loops

CPU_FLAGS = -march=native -mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq \
           -mfma -fopenmp -ffast-math -funroll-loops -ftree-vectorize \
           -fomit-frame-pointer -pthread

NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 \
          --use_fast_math --default-stream per-thread \
          -Xcompiler="${CPU_FLAGS}" \
          -maxrregcount=64 \
          -dlto

HIPCCFLAGS = -std=c++11 -O3 --offload-arch=gfx90a \
             -ffast-math -fopenmp \
             ${CPU_FLAGS}

LDFLAGS = -lm -fopenmp

LDFLAGS = -lm
EXES = seq

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cpp
	$(CXX) $(CXXFLAGS) -o $@ $?

gpu: gpu.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?
