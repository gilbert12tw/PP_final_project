#include <cstring>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#define max(a, b) (a > b ? a : b)
#define swap(a, b)    \
    {                 \
        int *tmp = a; \
        a = b;        \
        b = tmp;      \
    }

const int INF = ((1 << 30) - 1);
const int V = 50010;
void input(char *inFileName);
void output(char *outFileName);

void DP();
__global__ void DP_kernel(int *dp_table0, int *dp_table1, int weight, int value);

struct Item {
    int weight, value;
};

int n, m;
static Item *item;
static int *dp_table;

int main(int argc, char *argv[]) {
    input(argv[1]);
    DP();
    output(argv[2]);
    return 0;
}

void input(char *infile) {
    FILE *file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    item = (Item *)malloc(n * 2 * sizeof(int));
    fread(item, sizeof(int), n * 2, file);
    fclose(file);
}

void output(char *outFileName) {
    FILE *outfile = fopen(outFileName, "w");
    fwrite(&dp_table[m], sizeof(int), 1, outfile);
    fclose(outfile);
    cudaFreeHost(dp_table);
}

void DP() {
    int *dp_table0_dev, *dp_table1_dev;
    int const m_pad = ((m + 1) + 1024 - 1) / 1024 * 1024;
    cudaMalloc((void **)&dp_table0_dev, m_pad * sizeof(int));
    cudaMalloc((void **)&dp_table1_dev, m_pad * sizeof(int));
    for (int i = 0; i < n; i++) {
        DP_kernel<<<m_pad / 1024, 1024>>>(dp_table0_dev, dp_table1_dev, item[i].weight, item[i].value);
        swap(dp_table0_dev, dp_table1_dev);
    }

    cudaMallocHost((void **)&dp_table, (m + 1) * sizeof(int));
    cudaMemcpy(dp_table, dp_table0_dev, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dp_table0_dev);
    cudaFree(dp_table1_dev);
    free(item);
}

__global__ void DP_kernel(int *dp_table0, int *dp_table1, int weight, int value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int val = dp_table0[i];
    if (i >= weight) 
        val = max(val, dp_table0[i - weight] + value);
    dp_table1[i] = val;
}
