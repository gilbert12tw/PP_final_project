#include <algorithm>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 1024
#define CHUNK_SIZE 4096

#define max(a, b) (a > b ? a : b)
#define INF ((1 << 30) - 1)

// #define NDEBUG
#include <cassert>

struct Item {
    int weight, value;
};

void input(char *infile, int &n, int &m, Item *&items);
void output(char *outFileName, int result);

static inline bool compareByValue(const Item &a, const Item &b) {
    return a.value == b.value ? a.weight < b.weight : a.value < b.value;
}
static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

void processGroups(Item *items, int n,
                   int *&group_counts, int *&group_values,
                   int &num_groups);

int *mckp(Item *items, int *group_counts, int *group_values,
          int m, int num_groups);
__global__ void mckp_kernel(int *dp_prev, int *dp_curr,
                            int *group_weights, int group_value);

int main(int argc, char *argv[]) {
#ifndef NDEBUG
    if (argc != 3) {
        printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount != 2) {
        printf("Error: This program requires two GPUs\n");
        return 1;
    }
#endif

    int n, m;
    Item *items = NULL;

    input(argv[1], n, m, items);

    int *group_counts = NULL;
    int *group_values = NULL;
    int num_groups;

    processGroups(items, n, group_counts, group_values, num_groups);

    int num_groups_mid = num_groups / 2;
    int num_items_mid = 0;
    for (int i = 0; i < num_groups_mid; i++)
        num_items_mid += group_counts[i];

    int *dp[2];
#pragma omp parallel for
    for (int i = 0; i < 2; ++i) {
        cudaSetDevice(i);
        dp[i] = mckp(items + i * num_items_mid,
                     group_counts + i * num_groups_mid,
                     group_values + i * num_groups_mid,
                     m, i ? num_groups - num_groups_mid : num_groups_mid);
    }

#pragma unroll(2)
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    int ans = 0;
    for (int i = 0; i <= m; i++)
        ans = max(ans, dp[0][i] + dp[1][m - i]);

    free(group_counts);
    free(group_values);
    free(items);

    output(argv[2], ans);

    free(dp[0]);
    free(dp[1]);

    return 0;
}

int *mckp(Item *items, int *group_counts, int *group_values,
          int m, int num_groups) {
    int const m_pad = ceil_div(m + 1, CHUNK_SIZE) * CHUNK_SIZE;
    int *d_dp_prev = NULL, *d_dp_curr = NULL;
    cudaMalloc((void **)&d_dp_prev, m_pad * sizeof(int));
    cudaMalloc((void **)&d_dp_curr, m_pad * sizeof(int));

    // cudaMemset(d_dp_prev, 0, m_pad * sizeof(int));
    // cudaMemset(d_dp_curr, 0, m_pad * sizeof(int));

    int numBlocks = m_pad / BLOCK_SIZE;

    int curr_pos = 0;
    int *d_group_weights = NULL;
    cudaMalloc((void **)&d_group_weights, CHUNK_SIZE * sizeof(int));
    int *group_weights = (int *)malloc(CHUNK_SIZE * sizeof(int));
    for (int g = 0; g < num_groups; g++) {
#pragma omp parallel for
        for (int i = 0; i < CHUNK_SIZE; i++)
            group_weights[i] = (i < group_counts[g] ? items[curr_pos + i].weight : INF);

        cudaMemcpy(d_group_weights, group_weights,
                   CHUNK_SIZE * sizeof(int), cudaMemcpyHostToDevice);

        mckp_kernel<<<numBlocks, BLOCK_SIZE>>>(d_dp_prev, d_dp_curr,
                                               d_group_weights, group_values[g]);

        int *temp = d_dp_prev;
        d_dp_prev = d_dp_curr;
        d_dp_curr = temp;

        curr_pos += group_counts[g];
    }
    cudaFree(d_group_weights);
    free(group_weights);

    int *result = (int *)malloc((m + 1) * sizeof(int));
    cudaMemcpy(result, d_dp_prev, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_dp_prev);
    cudaFree(d_dp_curr);

    return result;
}
__global__ void mckp_kernel(int *dp_prev, int *dp_curr,
                            int *group_weights, int group_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int shared_weights[CHUNK_SIZE];

#pragma unroll(4096 / 1024)
    for (int i = threadIdx.x; i < CHUNK_SIZE; i += blockDim.x)
        shared_weights[i] = group_weights[i];
    __syncthreads();

    int maxVal = dp_prev[idx];
    int sumW = 0;

    for (int k = 0; k < CHUNK_SIZE; k++) {
        sumW += shared_weights[k];
        if (sumW > idx) break;

        maxVal = max(maxVal, dp_prev[idx - sumW] + group_value * (k + 1));
    }

    dp_curr[idx] = maxVal;
}

void processGroups(Item *items, int n,
                   int *&group_counts, int *&group_values,
                   int &num_groups) {
    std::sort(items, items + n, compareByValue);

    num_groups = 1;
    int curr_value = items[0].value;
    int g_cnt = 1;
    for (int i = 1; i < n; i++) {
        if (items[i].value != curr_value || g_cnt >= CHUNK_SIZE) {
            num_groups++;
            curr_value = items[i].value;
            g_cnt = 1;
        } else {
            ++g_cnt;
        }
    }

    group_counts = (int *)malloc(num_groups * sizeof(int));
    group_values = (int *)malloc(num_groups * sizeof(int));
    memset(group_counts, 0, num_groups * sizeof(int));

    int group_idx = 0;
    curr_value = items[0].value;
    group_counts[0] = 1;
    group_values[0] = curr_value;

    for (int i = 1; i < n; i++) {
        if (items[i].value != curr_value || group_counts[group_idx] >= CHUNK_SIZE) {
            group_idx++;
            curr_value = items[i].value;
            group_values[group_idx] = curr_value;
            group_counts[group_idx] = 1;
        } else {
            group_counts[group_idx]++;
        }
    }
}

void input(char *infile, int &n, int &m, Item *&items) {
    FILE *file = fopen(infile, "rb");
    if (!file) {
        printf("Error: Cannot open input file\n");
        exit(1);
    }

    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    items = (Item *)malloc(n * sizeof(Item));

    fread(items, sizeof(int), 2 * n, file);

    fclose(file);
}
void output(char *outFileName, int result) {
    FILE *outfile = fopen(outFileName, "w");
    if (!outfile) {
        printf("Error: Cannot open output file\n");
        exit(1);
    }

    fwrite(&result, sizeof(int), 1, outfile);
    fclose(outfile);
}
