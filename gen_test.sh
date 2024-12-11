#!/bin/bash

# Test cases configuration
declare -A test_cases=(
    # Format: [testcase_number]="n m"
    # Small test cases
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
)

# Generate test cases and run answer program
for test_num in "${!test_cases[@]}"; do
    params=(${test_cases[$test_num]})
    n=${params[0]}
    m=${params[1]}
    
    # Generate input
    in_file="testcases/${test_num}.in"
    echo "Generating test case ${test_num}: n=${n}, m=${m}"
    ./utilities/gen "$in_file" "$n" "$m"
    
    # Generate answer
    out_file="testcases/${test_num}.out"
    echo "Generating answer for test case ${test_num}"
    ./seq "$in_file" "$out_file"
    
    # Print progress
    echo "Completed test case ${test_num}"
    echo "----------------------------------------"
done

echo "All test cases and answers generated successfully."
echo "Test cases are stored as XX.in and XX.out in the testcases directory."

# Optional: Verify if all files exist
echo -e "\nVerifying generated files:"
for test_num in "${!test_cases[@]}"; do
    printf "Test case %s: " "$test_num"
    if [ -f "testcases/${test_num}.in" ] && [ -f "testcases/${test_num}.out" ]; then
        echo "OK"
    else
        echo "Missing files!"
    fi
done
