set -xe
CC=g++
$CC -Wall -Wextra -Wshadow -Wpedantic -fopenmp -Wno-unused-function -std=c++20 -O3 -march=native main.cpp -o simd
