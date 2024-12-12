#include <cstring>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#define BLOCK_SIZE 1024
#define max(a, b) (a > b ? a : b)

const int INF = ((1 << 30) - 1);
const int V = 50010;
void input(char *inFileName);
void output(char *outFileName);

void DP();
__global__ void DP_kernel(int *dp_table_prv, int *dp_table_cur, int weight, int value);

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
    int *d_table_prv, *d_table_cur;
    int const m_pad = ((m + 1) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    cudaMalloc((void **)&d_table_prv, m_pad * sizeof(int));
    cudaMalloc((void **)&d_table_cur, m_pad * sizeof(int));

    dim3 const block(BLOCK_SIZE);
    dim3 const grid(m_pad / BLOCK_SIZE);

    for (int i = 0; i < n; i++) {
        DP_kernel<<<grid, block>>>(d_table_prv, d_table_cur, item[i].weight, item[i].value);
        d_table_cur ^= d_table_prv;
        d_table_prv ^= d_table_cur;
    }

    cudaMallocHost((void **)&dp_table, (m + 1) * sizeof(int));
    cudaMemcpy(dp_table, d_table_prv, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_table_prv);
    cudaFree(d_table_cur);
    free(item);
}

__global__ void DP_kernel(int *dp_table_prv, int *dp_table_cur, int weight, int value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int val = dp_table_prv[i];
    if (i >= weight)
        val = max(val, dp_table_prv[i - weight] + value);
    dp_table_cur[i] = val;
}
