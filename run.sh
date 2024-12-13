#!/bin/bash

TEST=/home/pp24/pp24s039/PP_final_project/testcases

srun -N1 -n1 --gres=gpu:1 time ./$1 $TEST/$2.in tmp.out

echo "------------------------------------"
echo "Your answer:"
./utilities/reader tmp.out 

echo "Correct  answer:"
./utilities/reader $TEST/$2.out
