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
EXES = seq knapsack_gpu knapsack_mckp gpu

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

debug: LDFLAGS += -DDEBUG -g

seq: seq.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $?

knapsack_gpu: knapsack_gpu.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

knapsack_gpu_opt: knapsack_gpu_opt.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

gpu: gpu.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

gpu_dbg: gpu.cu
	$(NVCC) -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -g -G $(LDFLAGS) -o $@ $?

knapsack_mckp: knapsack_mckp.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

knapsack_mckp_by_weight: knapsack_mckp_by_weight.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

knapsack_mitm: knapsack_mitm.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?
