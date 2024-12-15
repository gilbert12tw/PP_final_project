#include <cstring>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

// const int INF = ((1 << 30) - 1);
// const int V = 50010;
void input(char *inFileName);
void output(char *outFileName);

void DP();
__global__ void DP_kernel(int *dp_prv, int *dp_cur, int weight, int value);

struct Item {
    int weight, value;
};

int n, m;
static Item *item;
int result;

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

    item = (Item *)malloc(n * sizeof(Item));
    fread(item, sizeof(Item), n, file);
    fclose(file);
}
void output(char *outFileName) {
    FILE *outfile = fopen(outFileName, "w");
    fwrite(&result, sizeof(int), 1, outfile);
    fclose(outfile);
}

void DP() {
    int *d_dp_prv, *d_dp_cur;
    int const m_pad = ((m + 1) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    dim3 const block(BLOCK_SIZE);
    dim3 const grid(m_pad / BLOCK_SIZE);
    cudaMalloc((void **)&d_dp_prv, m_pad * sizeof(int));
    cudaMalloc((void **)&d_dp_cur, m_pad * sizeof(int));

    for (int i = 0; i < n; i++) {
        DP_kernel<<<grid, block>>>(d_dp_prv, d_dp_cur, item[i].weight, item[i].value);
        int *temp = d_dp_prv;
        d_dp_prv = d_dp_cur;
        d_dp_cur = temp;
    }

    cudaMemcpy(&result, &d_dp_prv[m], sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_dp_prv);
    cudaFree(d_dp_cur);
    free(item);
}
__global__ void DP_kernel(int *dp_prv, int *dp_cur, int weight, int value) {
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    int val = dp_prv[i];
    if (i >= weight)
        val = max(val, dp_prv[i - weight] + value);
    dp_cur[i] = val;
}
