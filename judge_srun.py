#!/usr/bin/env python3
import os
import sys
import subprocess
import re

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def parse_time_output(stderr):
    """Parse the output of GNU time command"""
    try:
        # Extract information using regex
        # Match pattern like: 25.69user 0.00system 0:25.70elapsed 99%CPU (0avgtext+0avgdata 6024maxresident)k
        pattern = r'(\d+\.\d+)user\s+(\d+\.\d+)system\s+(\d+):(\d+\.\d+)elapsed\s+(\d+)%CPU\s+\((\d+)avgtext\+(\d+)avgdata\s+(\d+)maxresident\)k'
        match = re.search(pattern, stderr)
        
        if match:
            user_time = float(match.group(1))          # User time
            system_time = float(match.group(2))        # System time
            minutes = int(match.group(3))              # Elapsed minutes
            seconds = float(match.group(4))            # Elapsed seconds
            cpu_percent = int(match.group(5))          # CPU usage percentage
            max_resident = int(match.group(8))         # Maximum resident set size
            
            elapsed_time = minutes * 60 + seconds      # Total elapsed time in seconds
            
            return {
                'elapsed': elapsed_time,
                'user': user_time,
                'system': system_time,
                'cpu_percent': cpu_percent,
                'max_memory_kb': max_resident
            }
    except Exception as e:
        print(f"Error parsing time output: {e}")
    return None

def run_gpu_test(exe_path: str, num_gpus: int = 1):
    testcase_dir = "testcases"
    i = 1
    total_time = 0
    total_memory = 0
    total_cpu = 0
    passed = 0
    total = 0
    
    print("\n" + BOLD + "Running GPU tests..." + RESET + "\n")
    
    while True:
        in_file = os.path.join(testcase_dir, str(i) + ".in")
        ans_file = os.path.join(testcase_dir, str(i) + ".out")
        
        if not os.path.exists(in_file):
            break
            
        total += 1
        out_file = os.path.join(testcase_dir, "temp.out")
        
        try:
            cmd = [
                "srun",
                "-N1",
                "-n1",
                f"--gres=gpu:{num_gpus}",
                "time",  # Using system's time command
                "./" + exe_path,
                in_file,
                out_file
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse time command output
            perf_stats = parse_time_output(result.stderr)
            if perf_stats:
                elapsed = perf_stats['elapsed']
                total_time += elapsed
                total_memory += perf_stats['max_memory_kb']
                total_cpu += perf_stats['cpu_percent']
            else:
                elapsed = 0
            
            # Compare output with answer
            with open(out_file, 'rb') as f1, open(ans_file, 'rb') as f2:
                if f1.read() == f2.read():
                    passed += 1
                    status = GREEN + "Correct" + RESET
                else:
                    status = RED + "Wrong Answer" + RESET
            
            # Print detailed performance stats for each case
            print(f"Case {i:2d}: {status} {elapsed:.3f}s")
            if perf_stats:
                print(f"        CPU: {perf_stats['cpu_percent']}%, Memory: {perf_stats['max_memory_kb']}KB")
                print(f"        User: {perf_stats['user']:.3f}s, System: {perf_stats['system']:.3f}s")
            
            if os.path.exists(out_file):
                os.remove(out_file)
            
        except subprocess.TimeoutExpired:
            print(f"Case {i:2d}: {YELLOW}Time Limit Exceeded{RESET}")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else f"Return code: {e.returncode}"
            print(f"Case {i:2d}: {RED}Runtime Error{RESET} - {error_msg}")
        except Exception as e:
            print(f"Case {i:2d}: {RED}Error{RESET} - {str(e)}")
            if os.path.exists(out_file):
                os.remove(out_file)
            
        i += 1

    # Print summary
    print("\n" + BOLD + "Summary:" + RESET)
    print(f"Total cases: {total}")
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
        print(f"Total time: {total_time:.3f}s")
        print(f"Average time: {total_time/total:.3f}s")
        print(f"Average CPU usage: {total_cpu/total:.1f}%")
        print(f"Average memory usage: {total_memory/total:.0f}KB")

def main():
    if not(2 <= len(sys.argv) <= 3):
        print("Usage: python {} <executable> <gpu>".format(sys.argv[0]))
        sys.exit(1)
        
    if len(sys.argv) == 2:
        run_gpu_test(sys.argv[1])
    elif len(sys.argv) == 3:
        assert sys.argv[2].isdigit(), "GPU number must be an integer"
        assert 0 < int(sys.argv[2]) <= 2, "GPU number must be in [1, 2]"
        run_gpu_test(sys.argv[1], int(sys.argv[2]))

if __name__ == "__main__":
    main()
