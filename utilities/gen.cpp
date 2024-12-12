#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <cstring>

using namespace std;

// Constants for data generation
const int MAX_N = 1000000;  // Maximum number of items
const int MAX_M = 1000000;  // Maximum total capacity
const int MAX_V = 2000;   // Maximum value per item

void generate_test_data(const char* filename, int n, int m) {
    // Setup random number generator
    random_device rd;
    mt19937 gen(rd());
    
    // Define distributions
    uniform_int_distribution<int> weight_dist(1, m);
    uniform_int_distribution<int> value_dist(1, MAX_V);

    // Open output file in binary mode
    ofstream outFile(filename, ios::binary);
    if (!outFile) {
        cerr << "Error: Could not open file " << filename << endl;
        return;
    }

    // Write n and m
    outFile.write(reinterpret_cast<const char*>(&n), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&m), sizeof(int));

    // Generate and write item data
    for (int i = 0; i < n; i++) {
        int w1 = weight_dist(gen);
        int w2 = weight_dist(gen);
        int w = min(w1, w2);
        int v = value_dist(gen);
        
        // Write weight and value
        outFile.write(reinterpret_cast<const char*>(&w), sizeof(int));
        outFile.write(reinterpret_cast<const char*>(&v), sizeof(int));
    }

    outFile.close();
}

void print_file_content(const char* filename) {
    ifstream inFile(filename, ios::binary);
    if (!inFile) {
        cerr << "Error: Could not open file for reading" << endl;
        return;
    }

    // Read and print n and m
    int n, m;
    inFile.read(reinterpret_cast<char*>(&n), sizeof(int));
    inFile.read(reinterpret_cast<char*>(&m), sizeof(int));
    cout << "n = " << n << ", m = " << m << endl;
    
    // Read and print each item
    /*
    cout << "Items (weight, value):" << endl;
    for (int i = 0; i < n; i++) {
        int w, v;
        inFile.read(reinterpret_cast<char*>(&w), sizeof(int));
        inFile.read(reinterpret_cast<char*>(&v), sizeof(int));
        cout << i+1 << ": (" << w << ", " << v << ")" << endl;
    }
    */

    inFile.close();
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <output_file> <n> <m>" << endl;
        cerr << "n: number of items (1-" << MAX_N << ")" << endl;
        cerr << "m: knapsack capacity (1-" << MAX_M << ")" << endl;
        return 1;
    }

    // Parse command line arguments
    const char* filename = argv[1];
    int n = atoi(argv[2]);
    int m = atoi(argv[3]);

    // Validate input
    if (n <= 0 || n > MAX_N || m <= 0 || m > MAX_M) {
        cerr << "Error: Invalid input parameters" << endl;
        cerr << "n must be between 1 and " << MAX_N << endl;
        cerr << "m must be between 1 and " << MAX_M << endl;
        return 1;
    }

    // Generate test data
    generate_test_data(filename, n, m);
    
    // Print the generated data to verify
    cout << "Generated test data:" << endl;
    print_file_content(filename);

    return 0;
}
