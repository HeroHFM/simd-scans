#define NDEBUG

#include <functional>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <vector>
#include <limits>
#include <random>
#include <chrono>

#include <xmmintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

// #define VALIDATE

constexpr std::size_t SEED1 = 997;
constexpr std::size_t SEED2 = 88889872;

template <class /* Real */ R, std::size_t /* Size */ N, std::size_t /* Align */ A>
struct InternalNode {

	InternalNode() = default;
	~InternalNode() = default;

	// Fill with random values
	constexpr InternalNode(std::size_t n_keys, float min, float max)
	{
		assert(n_keys <= N);

		std::mt19937 gen(SEED1);
		std::uniform_real_distribution<R> dist(min, max);

		for (std::size_t i = 0; i < n_keys; keys[i++] = dist(gen));
		std::sort(keys, keys + n_keys);

		length = n_keys;
	}

	alignas(A) R keys[N] = { 0 };
	std::size_t length;
};

// Sequential Search Algorithms

static uint32_t sequential_search_1(const float array[], float target, std::size_t n)
{
	__m128i vsum = _mm_setzero_si128();

	for (std::size_t i = 0; i < n; i += 4)
    {
    	__m128 m = _mm_cmplt_ps(_mm_load_ps(&array[i]), _mm_set_ps1(target));
    	vsum = _mm_add_epi32(vsum, _mm_and_si128(_mm_set1_epi32(1), _mm_castps_si128(m)));
    }

    // Coalesce
    vsum = _mm_hadd_epi32(vsum, vsum);
    vsum = _mm_hadd_epi32(vsum, vsum);
    return _mm_cvtsi128_si32(vsum);
}

static uint32_t sequential_search_1ps(const float array[], float target, std::size_t n)
{
	__m128 vsum = _mm_setzero_ps();

	for (std::size_t i = 0; i < n; i += 4)
    {
    	__m128 m = _mm_cmplt_ps(_mm_load_ps(&array[i]), _mm_set_ps1(target));
    	vsum = _mm_add_ps(vsum, _mm_and_ps(_mm_set1_ps(1), m));
    }

    // Coalesce
    vsum = _mm_hadd_ps(vsum, vsum);
    vsum = _mm_hadd_ps(vsum, vsum);
    return _mm_cvtss_f32(vsum);
}

static uint32_t sequential_search_2(const float array[], float target, std::size_t n)
{
	__m128i vsum = _mm_setzero_si128();

	for (std::size_t i = 0; i < n; i += 4)
    {
    	__m128 m = _mm_cmplt_ps(_mm_load_ps(&array[i]), _mm_set_ps1(target));
    	vsum = _mm_add_epi32(vsum, _mm_and_si128(_mm_set1_epi32(1), _mm_castps_si128(m)));
    	// only need to check one
    	if (_mm_cvtsi128_si32(_mm_castps_si128(m)) == 0) { break; }
    }

    // Coalesce
    vsum = _mm_hadd_epi32(vsum, vsum);
    vsum = _mm_hadd_epi32(vsum, vsum);
    return _mm_cvtsi128_si32(vsum);
}

// Binary Search Algorithms

// Binary search implementation from the standard library
static uint32_t std_binary_search(const float array[], float target, std::size_t n)
	{ return std::lower_bound(array, array + n, target) - array; }

// Standard no-frills binary search implementation 
static uint32_t normal_binary_search(const float array[], float target, std::size_t n)
{
	uint32_t l = 0;
	uint32_t r = n;

	// Search
	while (l < r) {
		uint32_t c = std::midpoint(l, r);
		if (array[c] < target) { l = c + 1; } // Go right
		else                   { r = c;     } // Go left
	}

	return l;
}

static uint32_t hybrid_binary_search(const float array[], float target, std::size_t n, std::size_t lim)
{
	assert(lim % 4 == 0);
	uint32_t l = 0;
	uint32_t r = n / lim;

	// Search
	while (l < r) {
		uint32_t c = std::midpoint(l, r);
		if (array[std::min((c * lim) + lim - 1, n - 1)] < target) { l = c + 1; } // Go right
		else                                                      { r = c;     } // Go left
	}

	std::size_t pos = l * lim;
	return pos + sequential_search_1(array + pos, target, std::min(lim, n - pos));
}

static uint32_t simd_binary_search(const float array[], float target, std::size_t n)
{
	uint32_t l = 0;
	uint32_t r = n / 4;

	while (l < r) {
		uint32_t c = std::midpoint(l, r);
		
		__m128 m = _mm_cmplt_ps(_mm_load_ps(array + c * 4), _mm_set_ps1(target));
		if      (_mm_test_all_ones(_mm_castps_si128(m)))                       { l = c + 1; }
		else if (_mm_test_all_zeros(_mm_castps_si128(m), _mm_castps_si128(m))) { r = c;     }
		else {
			// Find the index
			__m128i vsum = _mm_and_si128(_mm_castps_si128(m), _mm_set1_epi32(1));
			vsum = _mm_hadd_epi32(vsum, vsum);
			vsum = _mm_hadd_epi32(vsum, vsum);
			return c * 4 + _mm_cvtsi128_si32(vsum);
		}
	}

	return l * 4;
}

// Hybrid binary search (switch to sequential at lim)
static uint32_t hybrid_simd_binary_search(const float array[], float target, std::size_t n, std::size_t lim)
{
	uint32_t l = 0;
	uint32_t r = n / 4;

	while (r - l > (lim / 4)) {
		uint32_t c = std::midpoint(l, r);
		
		__m128 m = _mm_cmplt_ps(_mm_load_ps(array + c * 4), _mm_set_ps1(target));
		if      (_mm_test_all_ones(_mm_castps_si128(m)))                       { l = c + 1; }
		else if (_mm_test_all_zeros(_mm_castps_si128(m), _mm_castps_si128(m))) { r = c;     }
		else {
			// Find the index
			__m128i vsum = _mm_and_si128(_mm_castps_si128(m), _mm_set1_epi32(1));
			vsum = _mm_hadd_epi32(vsum, vsum);
			vsum = _mm_hadd_epi32(vsum, vsum);
			return c * 4 + _mm_cvtsi128_si32(vsum);
		}
	}

	assert(l <= r);

	return (l * 4) + sequential_search_1(array + (l * 4), target, (r - l) * 4);
}

// Modified from: https://probablydance.com/2023/04/27/beautiful-branchless-binary-search/
// probably branchless
static uint32_t branchless_binary_search(const float array[], float target, std::size_t n)
{
	const float *cur = array;
	std::size_t step = std::bit_floor(n);

	// Correct the size if n is not a power of 2
	if (step != n && cur[step] < target) {
		if (n - step + 1 == 0) return n;
		step = std::bit_ceil(n - step + 1);
		cur += n - step;
	}

	for (step >>= 1; step; step >>= 1) {
		if (cur[step] < target) { cur += step; } // Compiles to cmov
	} 

	return (cur - array) + (*cur < target);
}

// Should be autovectorized to AVX2
static uint32_t sequential_search_scalar(const float array[], float target, std::size_t n)
{
	uint32_t c = 0;
	for (std::size_t i = 0; i < n; c += array[i++] < target);
    return c;
}

// Lim it a power of 2
static uint32_t branchless_hybrid_simd_binary_search(const float array[], float target, std::size_t n, std::size_t lim)
{
	const float *cur = array;
	std::size_t step = std::bit_floor(n);

	// Correct the size if n is not a power of 2
	if (step != n && cur[step] < target) {
		if (n - step + 1 == 0) { return n; }
		step = std::bit_ceil(n - step + 1);
		cur += n - step;
	}

	lim = std::min(lim, step);

	for (step >>= 4; step >= (lim / 8); step >>= 1) {
		__m256 m = _mm256_cmp_ps(_mm256_load_ps(cur + step * 8), _mm256_set1_ps(target), _CMP_LT_OQ);
		if (!_mm256_testz_si256(_mm256_castps_si256(m), _mm256_castps_si256(m)))
			{ cur += step * 8; }
	}

	return (cur - array) + sequential_search_scalar(cur, target, lim);
}


// Testing parameters

// Note: Interior nodes are at least half-full, but we will test
//       full range to be consistent with the paper.
const std::size_t MAX_SZ = 512; // Max # keys in interior node
const std::size_t MIN_SZ = 8;   // Min # keys in interior node

const std::size_t S = 8;        // SIMD Width
const std::size_t ALIGN  = 32;

const std::size_t NQUERIES = 10000;

const std::size_t NREPEAT = 15ul;

const std::size_t HYBRIDL = 128;

// Test a given algorithm
auto test_algorithm(auto & func, auto & queries, int nthreads = 1)
{
	std::vector<double> times;
	times.reserve(MAX_SZ / S);

	for (std::size_t n_items = MIN_SZ; n_items <= MAX_SZ; n_items += S) {

		InternalNode<float, MAX_SZ, ALIGN> node(n_items, 10, 190);

		std::size_t q[NQUERIES] = { 0 };

		auto begin = std::chrono::high_resolution_clock::now();

		for (std::size_t i = 0; i < NREPEAT; ++i) {

			std::size_t c = 0;
			#pragma omp parallel for schedule(static) num_threads(nthreads)
			for (const auto &query : queries) {
#ifdef VALIDATE
				std::size_t ref = std::lower_bound(node.keys, node.keys + node.length, query) - node.keys;
				q[c++] = func(node.keys, query, node.length);
				if (ref != q[c - 1]) {
					std::cout << "Validation failure(n=" << n_items
					          << ", query=" << query << ", exp=" << ref
					          << ", got=" << q[c - 1] << ")" << std::endl;
					exit(1);
				}
#else
				q[c++] = func(node.keys, query, node.length);
#endif
			}
		}

		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

		// Do something with the result
		std::cerr << std::accumulate(q, q + NQUERIES, 0ul) << "\n";

		times.push_back(duration / static_cast<double>(NREPEAT));
	}

	return times;
}

// Test a given algorithm
auto vary_algorithm(auto & func, auto n_items, auto & queries)
{
	std::vector<double> times;
	times.reserve(MAX_SZ / S);

	for (std::size_t L = MIN_SZ; L <= MAX_SZ; L += S) {

		InternalNode<float, MAX_SZ, ALIGN> node(n_items, 10, 190);

		std::size_t q[NQUERIES] = { 0 };

		auto begin = std::chrono::high_resolution_clock::now();

		for (std::size_t i = 0; i < NREPEAT; ++i) {
			std::size_t c = 0;
			for (const auto &query : queries) {
#ifdef VALIDATE
				std::size_t ref = std::lower_bound(node.keys, node.keys + node.length, query) - node.keys;
				q[c++] = func(node.keys, query, node.length, L);
				if (ref != q[c - 1]) {
					std::cout << "Validation failure(n=" << n_items
					          << ", query=" << query << ", exp=" << ref
					          << ", got=" << q[c -1 ] << ")" << std::endl;
					exit(1);
				}
#else
				q[c++] = func(node.keys, query, node.length, L);
#endif
			}
		}

		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

		// Do something with the result
		std::cerr << std::accumulate(q, q + NQUERIES, 0ul) << "\n";

		times.push_back(duration / static_cast<double>(NREPEAT));
	}

	return times;
}

// Dump vector of times to stdout as a CSV row
void dump_csv_row(const char * name, const auto & vec)
{
	if (name != nullptr) { std::cout << name << ", "; }
	for (auto it = vec.cbegin(); it != vec.cend(); ++it) {
		if (it != vec.cbegin()) std::cout << ", ";
		std::cout << *it;
	}
	std::cout << std::endl;
}

#define OP 0 // 0 = all, 1 = threading, 2 = L-value, 3 = single test (for perf)

// Cite: https://www.cs.columbia.edu/~kar/pubsk/simd.pdf
int main()
{
#if OP!=3
	// Ensure correct floating point type
	static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 FP");

	// Create random queries
	std::vector<float> queries;
	queries.reserve(NQUERIES);
	
	std::mt19937 gen(SEED2);
	std::uniform_real_distribution<float> dist(0, 200);

	for (std::size_t i = 0; i <= NQUERIES; ++i) { queries.push_back(dist(gen)); }


	// Write out N values
	std::vector<std::size_t> val;
	val.reserve(MAX_SZ / S);
	for (std::size_t n_items = MIN_SZ; n_items <= MAX_SZ; n_items += S) val.push_back(n_items);
	dump_csv_row(nullptr, val);
#endif

	// Collect statistics

#if OP==0 // Profile All

	// Sequential
	dump_csv_row("sequential1", test_algorithm(sequential_search_1, queries));
	dump_csv_row("sequential2", test_algorithm(sequential_search_2, queries));
	// dump_csv_row("sequentialf", test_algorithm(sequential_search_scalar, queries));
	
	// Binary Search
	dump_csv_row("normal_binary", test_algorithm(normal_binary_search, queries));
	dump_csv_row("std::lower_bound", test_algorithm(std_binary_search, queries));
	dump_csv_row("simd_binary", test_algorithm(simd_binary_search, queries));

	auto h1 = [](const float array[], float target, std::size_t n){
		return hybrid_binary_search(array, target, n, HYBRIDL);
	};
	dump_csv_row("hybrid_binary", test_algorithm(h1, queries));

	auto h2 = [](const float array[], float target, std::size_t n){
		return hybrid_simd_binary_search(array, target, n, HYBRIDL);
	};
	dump_csv_row("hybrid_simd_binary", test_algorithm(h2, queries));

	// Branchless

	/*
	dump_csv_row("branchless_binary", test_algorithm(branchless_binary_search, queries));

	auto b = [](const float array[], float target, std::size_t n){
		return branchless_hybrid_simd_binary_search(array, target, n, 64);
	};

	dump_csv_row("branchless_hybrid_simd_binary", test_algorithm(b, queries));

	*/

	// dump_csv_row("branchless_hybrid_simd_binary_t8",
	// 	test_algorithm(b, queries, 8));

	// dump_csv_row("branchless_simd_binary",
	// 	test_algorithm(branchless_simd_binary_search, queries)
	// );

	// dump_csv_row("branchless_simd_binary2",
	// 	test_algorithm(branchless_simd_binary_search_2, queries)
	// );
	
#elif OP==1 // Threading
	auto h = [](const float array[], float target, std::size_t n){
		return hybrid_simd_binary_search(array, target, n, HYBRIDL);
	};
	dump_csv_row("hybrid_simd_binary_t1",  test_algorithm(h, queries,  1));
	dump_csv_row("hybrid_simd_binary_t2",  test_algorithm(h, queries,  2));
	dump_csv_row("hybrid_simd_binary_t4",  test_algorithm(h, queries,  4));
	dump_csv_row("hybrid_simd_binary_t8",  test_algorithm(h, queries,  8));
	dump_csv_row("hybrid_simd_binary_t12", test_algorithm(h, queries, 12));
#elif OP==2 // L-value
	dump_csv_row("hybrid_binary", vary_algorithm(hybrid_binary_search, 512, queries));
	dump_csv_row("hybrid_simd_binary", vary_algorithm(hybrid_simd_binary_search, 512, queries));
	// dump_csv_row("branchless_hybrid_simd_binary", vary_algorithm(hybrid_simd_binary_search, 512, queries));
#elif OP == 3 // Run single instance
	#include "random_queries.h"

	std::size_t q[NQUERIES] = { 0 };
	
	for (std::size_t c = 0; c < NQUERIES; ++c) {
		q[c] = sequential_search_1ps(keys, queries[c], 512);
	}

	std::cerr << std::accumulate(q, q + NQUERIES, 0ul) << "\n";
#endif
}
