#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define max(a, b) (a > b ? a : b)
const int INF = ((1 << 30) - 1);
const int V = 50010;

struct Item {
    int weight, value;
};

__global__ void dp_kernel(int* dp_prev, int* dp_curr, int m, int weight, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx <= m) {
        if (idx >= weight) {
            dp_curr[idx] = max(dp_prev[idx], dp_prev[idx - weight] + value);
        } else {
            dp_curr[idx] = dp_prev[idx];
        }
    }
}

void input(char* inFileName, int& n, int& m, Item*& items);
void output(char* outFileName, int* result, int m);

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    // Variables for host
    int n, m;
    Item* items;
    
    // Read input
    input(argv[1], n, m, items);
    
    // Allocate device memory for DP tables
    int *d_dp_prev, *d_dp_curr;
    cudaMalloc((void**)&d_dp_prev, (m + 1) * sizeof(int));
    cudaMalloc((void**)&d_dp_curr, (m + 1) * sizeof(int));
    
    // Initialize DP table
    cudaMemset(d_dp_prev, 0, (m + 1) * sizeof(int));
    
    // Calculate grid and block dimensions
    int numBlocks = (m + BLOCK_SIZE) / BLOCK_SIZE;
    
    // Main DP loop
    for (int i = 0; i < n; i++) {
        dp_kernel<<<numBlocks, BLOCK_SIZE>>>(d_dp_prev, d_dp_curr, m, 
                                           items[i].weight, items[i].value);
        
        // Swap pointers for next iteration
        int *temp = d_dp_prev;
        d_dp_prev = d_dp_curr;
        d_dp_curr = temp;
    }
    
    // Allocate host memory for result
    int* result = (int*)malloc((m + 1) * sizeof(int));
    
    // Copy result back to host
    cudaMemcpy(result, d_dp_prev, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Output result
    output(argv[2], result, m);
    
    // Cleanup
    cudaFree(d_dp_prev);
    cudaFree(d_dp_curr);
    free(items);
    free(result);
    
    return 0;
}

void input(char* infile, int& n, int& m, Item*& items) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    
    items = (Item*)malloc(n * sizeof(Item));
    fread(items, sizeof(Item), n, file);
    
    fclose(file);
}

void output(char* outFileName, int* result, int m) {
    FILE* outfile = fopen(outFileName, "w");
    int ans = 0;
    for (int i = 1; i <= m; i++) {
        ans = max(ans, result[i]);
    }
    fwrite(&ans, sizeof(int), 1, outfile);
    fclose(outfile);
}
