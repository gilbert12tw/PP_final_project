#include <algorithm>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 1024
#define CHUNK_SIZE 4096
#define max(a, b) (a > b ? a : b)
#define INF ((1 << 30) - 1)

struct Item {
    int weight, value;
};

static inline bool compareByValue(const Item &a, const Item &b) {
    if (a.value == b.value) {
        return a.weight < b.weight;
    }
    return a.value < b.value;
}


static inline bool compareByWeight(const Item &a, const Item &b) {
    if (a.weight == b.weight) {
        return a.value > b.value;
    }
    return a.weight < b.weight;
}

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

__global__ void mckp_kernel(int *dp_prev, int *dp_curr, int *group_values, int group_weights) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int shared_values[CHUNK_SIZE];

#pragma unroll(4096 / 1024)
    for (int i = threadIdx.x; i < CHUNK_SIZE; i += blockDim.x)
        shared_values[i] = group_values[i];
    __syncthreads();

    int maxVal = dp_prev[idx];
    int sumV = 0;

    for (int k = 0; k < CHUNK_SIZE; k++) {
        sumV += shared_values[k];
        if (group_weights * (k + 1) > idx) break;

        maxVal = max(maxVal, dp_prev[idx - group_weights * (k + 1)] + sumV);
    }

    dp_curr[idx] = maxVal;
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

void processGroups(Item *items, int n, int *&group_counts, int *&unique_weight, int &num_groups) {
    std::sort(items, items + n, compareByWeight);

    num_groups = 1;
    int curr_weight = items[0].weight;
    int g_cnt = 1;
    for (int i = 1; i < n; i++) {
        if (items[i].weight != curr_weight || g_cnt >= CHUNK_SIZE) {
            num_groups++;
            curr_weight = items[i].weight;
            g_cnt = 1;
        } else {
            ++g_cnt;
        }
    }

    group_counts = (int *)malloc(num_groups * sizeof(int));
    unique_weight = (int *)malloc(num_groups * sizeof(int));
    memset(group_counts, 0, num_groups * sizeof(int));

    int group_idx = 0;
    curr_weight = items[0].weight;
    group_counts[0] = 1;
    unique_weight[0] = curr_weight;

    for (int i = 1; i < n; i++) {
        if (items[i].weight != curr_weight || group_counts[group_idx] >= CHUNK_SIZE) {
            group_idx++;
            curr_weight = items[i].weight;
            unique_weight[group_idx] = curr_weight;
            group_counts[group_idx] = 1;
        } else {
            group_counts[group_idx]++;
        }
    }
}

void output(char *outFileName, int result, int m) {
    FILE *outfile = fopen(outFileName, "w");
    if (!outfile) {
        printf("Error: Cannot open output file\n");
        exit(1);
    }

    fwrite(&result, sizeof(int), 1, outfile);
    fclose(outfile);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    int n, m;
    Item *items = NULL;

    input(argv[1], n, m, items);

    int *group_counts = NULL;
    int *unique_weight = NULL;
    int num_groups;

    processGroups(items, n, group_counts, unique_weight, num_groups);

    int const m_pad = ceil_div(m + 1, CHUNK_SIZE) * CHUNK_SIZE;
    int *d_dp_prev = NULL, *d_dp_curr = NULL;
    cudaMalloc((void **)&d_dp_prev, m_pad * sizeof(int));
    cudaMalloc((void **)&d_dp_curr, m_pad * sizeof(int));

    // cudaMemset(d_dp_prev, 0, m_pad * sizeof(int));
    // cudaMemset(d_dp_curr, 0, m_pad * sizeof(int));

    int numBlocks = m_pad / BLOCK_SIZE;

    int curr_pos = 0;
    int *d_group_values = NULL;
    cudaMalloc((void **)&d_group_values, CHUNK_SIZE * sizeof(int));
    int *group_values = NULL;
    cudaMallocHost(&group_values, CHUNK_SIZE * sizeof(int));
    printf("num_groups = %d\n", num_groups);
    for (int g = 0; g < num_groups; g++) {
        #pragma omp parallel for
        for (int i = 0; i < CHUNK_SIZE; i++)
            group_values[i] = (i < group_counts[g] ? items[curr_pos + i].value : 0);

        cudaMemcpy(d_group_values, group_values,
                   CHUNK_SIZE * sizeof(int), cudaMemcpyHostToDevice);

        mckp_kernel<<<numBlocks, BLOCK_SIZE>>>(d_dp_prev, d_dp_curr, d_group_values, unique_weight[g]);

        int *temp = d_dp_prev;
        d_dp_prev = d_dp_curr;
        d_dp_curr = temp;

        curr_pos += group_counts[g];
    }
    cudaFree(d_group_values);
    cudaFreeHost(group_values);

    int result;
    cudaMemcpy(&result, &d_dp_prev[m], sizeof(int), cudaMemcpyDeviceToHost);

    output(argv[2], result, m);

    cudaFree(d_dp_prev);
    cudaFree(d_dp_curr);
    free(group_counts);
    free(unique_weight);
    free(items);

    return 0;
}
