# Readme

### Generate Test Cases

```bash
./utilities/gen <output_file> <n> <m>
```

- `n` total number of items
- `m` max weight of the knapsack

### Run Solution

```bash
./seq <input_file> <output_file>
```

### Read Binary Files

```bash
./utilities/reader <binary_file>
```

### Run Tests

```bash
python3 judge.py exe
```

## Test Cases

Each test case consists of:

- `X.in`: Input file in binary format
- `X.out`: Expected output file in binary format

## File Format

### Input Format (*.in)

- Binary file containing integers (4 bytes each)
- First two integers: n, m
- Followed by data specific to the problem

### Output Format (*.out)

- Binary file containing integers (4 bytes each)
- Contains only one integer represent the maximum weight that can fit in the knapsack

## Utilities

### Generator (gen)

- Located in `utilities/gen.cpp`
- Used to generate test cases
- Creates binary input files

### Reader (reader)

- Located in `utilities/reader.cpp`
- Utility to read and display binary files
- Helps in debugging and verifying test cases

### testcases parameter
    ["1"]="10 50"
    ["2"]="20 100"
    ["3"]="50 200"

    # Medium test cases
    ["4"]="100 500"
    ["5"]="200 1000"
    ["6"]="500 2000"

    # Large test cases
    ["7"]="1000 5000"
    ["8"]="2000 8000"
    ["9"]="5000 10000"
    ["10"]="10000 10000"

    # for GPU
    ["11"]="100000 10000"
    ["12"]="100000 100000"
    ["13"]="100000 1000000"
    ["14"]="1000000 100000"
    ["15"]="1000000 1000000"

