#!/usr/bin/env python3
import os
import sys
import time
import subprocess

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def run_test(exe_path):
    # Get all test cases
    testcase_dir = "testcases"
    i = 1
    total_time = 0
    passed = 0
    total = 0

    print("\n" + BOLD + "Running tests..." + RESET + "\n")
    
    while True:
        in_file = os.path.join(testcase_dir, str(i) + ".in")
        ans_file = os.path.join(testcase_dir, str(i) + ".out")
        
        # Stop if test case doesn't exist
        if not os.path.exists(in_file):
            break
            
        total += 1
        # Create temp output file name
        out_file = os.path.join(testcase_dir, "temp.out")
        
        # Run program with timing
        try:
            start_time = time.time()
            result = subprocess.run(
                ["srun", "-N1", "-n1", "--gres=gpu:1", "./" + exe_path, in_file, out_file], 
                capture_output=True,
                text=True,
                check=True
            )
            end_time = time.time()
            elapsed = end_time - start_time
            total_time += elapsed
            
            # Compare output with answer
            with open(out_file, 'rb') as f1, open(ans_file, 'rb') as f2:
                if f1.read() == f2.read():
                    passed += 1
                    status = GREEN + "Correct" + RESET
                else:
                    status = RED + "Wrong Answer" + RESET
                
            print("Case {:2d}: {} {:.3f}s".format(i, status, elapsed))
                    
            # Clean up
            if os.path.exists(out_file):
                os.remove(out_file)
            
        except subprocess.TimeoutExpired:
            print("Case {:2d}: {}Time Limit Exceeded{} {:.3f}s".format(
                i, YELLOW, RESET, elapsed))
        except subprocess.CalledProcessError as e:
            print("Case {:2d}: {}Runtime Error{} (Return code: {})".format(
                i, RED, RESET, e.returncode))
        except Exception as e:
            print("Case {:2d}: {}Error{} - {}".format(i, RED, RESET, str(e)))
            if os.path.exists(out_file):
                os.remove(out_file)
            
        i += 1

    # Print summary
    print("\n" + BOLD + "Summary:" + RESET)
    print("Total cases: {}".format(total))
    print("Passed: {}{}/{}{}".format(
        GREEN if passed == total else RED,
        passed, total,
        RESET
    ))
    if total > 0:
        print("Pass rate: {}{:.1f}%{}".format(
            GREEN if passed == total else RED,
            (passed/total)*100,
            RESET
        ))
        print("Total time: {:.3f}s".format(total_time))
        print("Average time: {:.3f}s".format(total_time/total))

def main():
    if len(sys.argv) != 2:
        print("Usage: python {} <executable>".format(sys.argv[0]))
        sys.exit(1)
        
    run_test(sys.argv[1])

if __name__ == "__main__":
    main()
