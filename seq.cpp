#include <stdio.h>
#include <stdlib.h>

#define max(a, b) (a > b ? a : b)

const int INF = ((1 << 30) - 1);
const int V = 50010;
void input(char* inFileName);
void output(char* outFileName);

void DP();

struct Item {
    int weight, value;
};

int n, m;
static Item *item;
int *dp_table;

int main(int argc, char* argv[]) {
    input(argv[1]);
    DP();
    output(argv[2]);
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    item = (Item*)malloc(m * 2 * sizeof(int));
    fread(item, sizeof(int), m * 2, file);
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    int ans = 0;
    for (int i = 1; i <= m; i++) ans = max(ans, dp_table[i]);
    fwrite(&ans, sizeof(int), 1, outfile);
    fclose(outfile);
}

void DP() {
    dp_table = (int*)malloc((m + 1) * sizeof(int));
    for (int i = 0; i < n; i++) {
        int w = item[i].weight;
        int v = item[i].value;
        for (int j = m; j >= w; j--) {
            dp_table[j] = max(dp_table[j], dp_table[j - w] + v);
        }
    }
}
